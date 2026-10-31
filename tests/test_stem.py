"""The P2 conv stem, input adapter, WST stem, multi-layer fusion and both RoI heads, end to end (CPU only)."""
import pytest
import torch
from torch import nn

from sarbench.heads import ConvFCBoxHead
from sarbench.model import ConvStem, InputAdapter, LayerFusion, MultiTaskModel, WSTStem


def test_conv_stem_has_stride_four_and_the_requested_channels():
    stem = ConvStem(in_chans=3, channels=16, out_channels=64)
    out = stem(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 64, 16, 16)  # two stride-2 convs -> stride 4


def test_wst_stem_is_81_coefficients_at_stride_four():
    stem = WSTStem(out_channels=64, J=2, L=8)
    out = stem(torch.randn(1, 3, 512, 512))
    assert out.shape == (1, 64, 128, 128)  # 512 / 2**2 = 128, i.e. P2's resolution


def test_input_adapter_starts_as_a_no_op():
    adapter = InputAdapter(3)
    x = torch.rand(2, 3, 16, 16)
    normalized = (x - adapter.mean) / adapter.std
    assert torch.allclose(adapter(x), normalized, atol=1e-6)


class _Block(nn.Module):
    def forward(self, x):
        return x


class _StubBackbone(nn.Module):
    """The DINOv3 interface the model uses: embed_dim, in_chans, prefix tokens, blocks and a token forward."""

    def __init__(self, layers=4):
        super().__init__()
        self.embed_dim = 128
        self.in_chans = 3
        self.num_prefix_tokens = 1
        self.patch = nn.Conv2d(3, 128, 16, stride=16)  # a stand-in for a patch embedding
        self._blocks = nn.ModuleList([_Block() for _ in range(layers)])

    @property
    def blocks(self):
        return self._blocks

    def unfreeze_patch_embed(self):
        self.patch.requires_grad_(True)

    def forward(self, x):
        tokens = torch.cat([torch.zeros(x.shape[0], 1, self.embed_dim), self.patch(x).flatten(2).transpose(1, 2)], 1)
        for block in self._blocks:
            tokens = block(tokens)
        return tokens[:, 1:]


IMAGES = torch.randn(2, 3, 128, 128)
TARGETS = [{'boxes': torch.tensor([[10., 12., 28., 40.]]), 'labels': torch.tensor([1])},
           {'boxes': torch.tensor([[30., 30., 50., 60.]]), 'labels': torch.tensor([2])}]
CASCADE = {'num_stages': 3, 'share_head': False, 'hidden_dim': 64, 'fg_iou_thresholds': [0.4, 0.5, 0.6],
           'bg_iou_threshold': 0.4, 'stage_loss_weights': [0.5, 0.25, 0.25]}


def test_layer_fusion_preserves_shape_and_is_a_softmax_weighted_sum():
    fusion = LayerFusion(4)
    tokens = [torch.randn(2, 5, 8) for _ in range(4)]
    assert fusion(tokens).shape == (2, 5, 8)
    assert torch.isclose(fusion.weights.softmax(0).sum(), torch.tensor(1.0))


@pytest.mark.parametrize('head, params', [('standard', {}), ('cascade', CASCADE)])
def test_heads_train_with_giou_and_fusion_then_predict(head, params):
    model = MultiTaskModel(_StubBackbone(), num_classes=9, detail_stem=True, fusion_layers=[1, 2, 3],
                           giou_weight=1.0, head=head, head_params=params)
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert set(losses) == {'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg'}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.fusion.weights.grad is not None  # the fusion weights are trained
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(IMAGES)
    assert logits.shape == (2, 9) and losses == {} and len(detections) == 2


def test_unknown_head_is_rejected():
    with pytest.raises(ValueError):
        MultiTaskModel(_StubBackbone(), head='deform')


def test_model_with_detail_stem_runs():
    model = MultiTaskModel(_StubBackbone(), num_classes=9, detail_stem=True)
    assert isinstance(model.detail_stem, ConvStem)
    assert sum(p.numel() for p in model.detail_stem.parameters()) > 0
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 9) and losses == {} and len(detections) == 2


def test_model_with_input_adapter_and_wst_stem_runs():
    model = MultiTaskModel(_StubBackbone(), num_classes=9,
                           input_adapter=True, wst_stem=True)
    assert isinstance(model.input_adapter, InputAdapter) and isinstance(model.wst_stem, WSTStem)
    assert model.backbone.patch.weight.requires_grad  # the adapter unfreezes the patch embedding
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(torch.randn(2, 3, 512, 512))
    assert logits.shape == (2, 9) and len(detections) == 2


def test_detail_stem_width_and_extra_convs_keep_stride_four():
    stem = ConvStem(in_chans=3, channels=32, out_channels=64, extra_convs=2)
    assert stem(torch.randn(1, 3, 64, 64)).shape == (1, 64, 16, 16)
    plain = ConvStem(in_chans=3, channels=32, out_channels=64)
    assert set(plain.state_dict()) <= set(stem.state_dict())  # old checkpoints keep their layer names


@pytest.mark.parametrize('head', ['standard', 'cascade'])
def test_conv_box_head_trains_and_is_cheaper_than_the_mlp_head(head):
    params = {**(CASCADE if head == 'cascade' else {}), 'box_head': 'conv', 'num_convs': 2, 'reduce': 16}
    model = MultiTaskModel(_StubBackbone(), detail_stem=True, giou_weight=1.0, head=head, head_params=params)
    assert any(isinstance(m, ConvFCBoxHead) for m in model.roi_heads.modules())
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    sum(losses.values()).backward()
    mlp = MultiTaskModel(_StubBackbone(), head='standard')
    conv = MultiTaskModel(_StubBackbone(), head='standard', head_params={'box_head': 'conv'})
    count = lambda m: sum(p.numel() for p in m.roi_heads.parameters())
    assert count(conv) < count(mlp)


def test_rpn_and_postprocess_settings_reach_the_detector():
    model = MultiTaskModel(_StubBackbone(), rpn_fg_iou_thresh=0.6, rpn_bg_iou_thresh=0.25, score_thresh=0.001,
                           nms_thresh=0.6, detections_per_img=50)
    assert model.rpn.proposal_matcher.high_threshold == 0.6 and model.rpn.proposal_matcher.low_threshold == 0.25
    heads = model.roi_heads
    assert (heads.score_thresh, heads.nms_thresh, heads.detections_per_img) == (0.001, 0.6, 50)
    model.eval()
    with torch.no_grad():
        _, detections, _ = model(IMAGES)
    assert all(len(d['boxes']) <= 50 for d in detections)


def test_class_prior_rescoring_modes():
    from sarbench.heads import apply_class_prior
    logits = torch.randn(5, 10)
    prior = torch.rand(5, 9).softmax(-1)
    scores = logits.softmax(-1)
    soft0 = apply_class_prior(logits, prior, 'soft', power=0.0).softmax(-1)
    torch.testing.assert_close(soft0, scores)  # power 0: unchanged
    soft = apply_class_prior(logits, prior, 'soft', power=1.0).softmax(-1)
    torch.testing.assert_close(soft[:, 1:], scores[:, 1:] * prior)  # foreground scaled, background the remainder
    predicted = prior.argmax(1) + 1
    hard = apply_class_prior(logits, prior, 'hard').softmax(-1)
    rows = torch.arange(5)
    torch.testing.assert_close(hard[rows, predicted], scores[rows, predicted])
    assert (hard[:, 1:].sum(1) - hard[rows, predicted]).abs().max() < 1e-6  # every other class zeroed
    merge = apply_class_prior(logits, prior, 'merge').softmax(-1)
    torch.testing.assert_close(merge[rows, predicted], 1 - scores[:, 0])  # the whole foreground mass
    with pytest.raises(ValueError):
        apply_class_prior(logits, prior, 'sharpen')


def test_hard_rescoring_keeps_only_the_predicted_image_class():
    model = MultiTaskModel(_StubBackbone(), num_classes=9, rescore='hard', score_thresh=0.0)
    model.eval()
    with torch.no_grad():
        logits, detections, _ = model(IMAGES)
    for image_logits, det in zip(logits, detections):
        assert len(det['labels']) and (det['labels'] == image_logits.argmax() + 1).all()


def test_unknown_rescore_mode_is_rejected():
    with pytest.raises(ValueError):
        MultiTaskModel(_StubBackbone(), rescore='sharpen')


def test_nwd_is_lenient_for_shifted_tiny_boxes_and_max_keeps_iou_for_large_ones():
    from torchvision.ops import box_iou
    from sarbench.heads import box_similarity, normalized_wasserstein
    gt = torch.tensor([[100., 100., 108., 108.], [100., 100., 300., 300.]])
    boxes = gt + torch.tensor([2., 2., 2., 2.])  # both shifted by 2 px
    assert torch.allclose(normalized_wasserstein(gt, gt).diagonal(), torch.ones(2))
    iou, nwd = box_iou(gt, boxes).diagonal(), normalized_wasserstein(gt, boxes, 16.0).diagonal()
    assert iou[0] < 0.5 < 0.7 < nwd[0]  # tiny box: background by IoU, positive by NWD
    best = box_similarity('max', 16.0)(gt, boxes).diagonal()
    torch.testing.assert_close(best, torch.maximum(iou, nwd))
    with pytest.raises(ValueError):
        box_similarity('giou')


def test_box_similarity_reaches_the_rpn_and_the_roi_head_and_trains():
    model = MultiTaskModel(_StubBackbone(), box_similarity='nwd', nwd_constant=8.0, giou_weight=1.0)
    assert model.rpn.box_similarity is model.roi_heads.box_similarity
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert all(torch.isfinite(value) for value in losses.values())


def test_feature_pyramid_for_an_8px_vit_keeps_the_strides():
    from sarbench.model import SimpleFeaturePyramid
    neck = SimpleFeaturePyramid(dim=128, channels=64, patch_size=8)
    levels = neck(torch.randn(1, 128, 16, 16))  # a 128 px image at stride 8
    assert [tuple(f.shape[-2:]) for f in levels.values()] == [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)]
    with pytest.raises(ValueError):
        SimpleFeaturePyramid(dim=128, patch_size=14)


class _RecordingBlock(nn.Module):
    """A token-mixing-free block that accepts the keyword arguments DINOv3 passes (RoPE)."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, **kwargs):
        return x + self.linear(x)


def _adapter_backbone():
    backbone = _StubBackbone(layers=4)
    backbone._blocks = nn.ModuleList([_RecordingBlock(128) for _ in range(4)])
    return backbone.requires_grad_(False)


VIT_ADAPTER = {'dim': 32, 'heads': 4, 'points': 2, 'channels': 16, 'interactions': [[0, 1], [2, 3]]}


def test_vit_adapter_is_a_noop_on_the_backbone_at_init_and_trains():
    from sarbench.vit_adapter import InteractionBlock
    torch.manual_seed(0)
    plain = MultiTaskModel(_adapter_backbone(), detail_stem=False)
    adapted = MultiTaskModel(_adapter_backbone(), detail_stem=False, vit_adapter=VIT_ADAPTER)
    adapted.backbone.load_state_dict(plain.backbone.state_dict(), strict=False)  # wrapped: names differ
    for wrapped, block in zip(adapted.backbone.blocks, plain.backbone.blocks):
        inner = wrapped.block if isinstance(wrapped, InteractionBlock) else wrapped
        inner.load_state_dict(block.state_dict())
    assert sum(isinstance(b, InteractionBlock) for b in adapted.backbone.blocks) == 4
    adapted.vit_adapter.prepare(IMAGES)
    torch.testing.assert_close(adapted.backbone(IMAGES), plain.backbone(IMAGES))  # gamma = 0: injection is a no-op
    adapted.vit_adapter.outputs()
    # The adapter is the model's, not the backbone's: counted once, trained at the head learning rate.
    assert not any('vit_adapter' in name for name, _ in adapted.backbone.named_parameters())
    adapted.train()
    _, _, losses = adapted(IMAGES, TARGETS)
    sum(losses.values()).backward()
    adapter = adapted.vit_adapter
    assert adapter.injectors[0].gamma.grad is not None and adapter.extractors[1].attn.value_proj.weight.grad.abs().sum() > 0
    assert adapter.spatial_prior.stem[0].weight.grad.abs().sum() > 0


def test_vit_adapter_needs_prepare_before_the_backbone():
    model = MultiTaskModel(_adapter_backbone(), vit_adapter=VIT_ADAPTER)
    with pytest.raises(RuntimeError):
        model.backbone(IMAGES)


def test_deformable_attention_samples_at_the_reference_point():
    """One point per head with zero offset and a constant value map: the output is that value, projected."""
    from sarbench.vit_adapter import DeformableAttention, grid_centers
    attn = DeformableAttention(8, 4, 4, 4, heads=2, levels=1, points=1)
    with torch.no_grad():
        attn.sampling_offsets.bias.zero_()
        attn.value_proj.weight.copy_(torch.eye(4)), attn.output_proj.weight.copy_(torch.eye(4))
    value = torch.arange(4.).repeat(1, 6 * 6, 1)  # every pixel holds (0, 1, 2, 3)
    out = attn(torch.randn(1, 9, 8), grid_centers(3, 3), value, [(6, 6)])
    torch.testing.assert_close(out, torch.arange(4.).expand(1, 9, 4))


def test_box_quality_ranking_and_paired_iou():
    from sarbench.heads import apply_box_quality, paired_iou
    logits = torch.randn(4, 10)
    quality = torch.rand(4, 9)
    torch.testing.assert_close(apply_box_quality(logits, quality, alpha=0.0).softmax(-1), logits.softmax(-1))
    ranked = apply_box_quality(logits, quality, alpha=1.0).softmax(-1)
    torch.testing.assert_close(ranked[:, 1:], logits.softmax(-1)[:, 1:] * quality)
    boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]])
    torch.testing.assert_close(paired_iou(boxes, torch.tensor([[0., 0., 10., 10.], [5., 0., 15., 10.]])),
                               torch.tensor([1.0, 1 / 3]))


def test_iou_aware_head_trains_its_branch_and_ranks_by_it():
    model = MultiTaskModel(_StubBackbone(), giou_weight=0.0, iou_aware=True, iou_alpha=1.0, score_thresh=0.0)
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert 'loss_iou' in losses and torch.isfinite(losses['loss_iou'])
    sum(losses.values()).backward()
    assert model.roi_heads.iou_predictor.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        _, ranked, _ = model(IMAGES)
        model.roi_heads.iou_alpha = 0.0
        _, plain, _ = model(IMAGES)
    # a predicted IoU in (0, 1) can only lower a score; alpha 0 leaves the class scores
    assert ranked[0]['scores'].max() <= plain[0]['scores'].max() + 1e-6


def test_iou_aware_is_standard_head_only():
    with pytest.raises(ValueError):
        MultiTaskModel(_StubBackbone(), head='cascade', head_params=CASCADE, iou_aware=True)


def test_p2_heatmap_targets_peak_at_each_centre_and_the_loss_trains_only_in_training():
    from sarbench.model import CenterHeatmapHead
    head = CenterHeatmapHead(channels=256, num_classes=9)
    targets = [{'boxes': torch.tensor([[8., 8., 16., 16.], [40., 40., 120., 100.]]), 'labels': torch.tensor([3, 3])},
               {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.int64)}]
    heatmaps = head.targets(targets, 9, 32, 32, 'cpu')  # a 128 px image at stride 4
    assert heatmaps.shape == (2, 9, 32, 32) and heatmaps[1].sum() == 0
    assert heatmaps[0, 2, 3, 3] == 1 and heatmaps[0, 2, 17, 20] == 1  # centres (12, 12) and (80, 70) px
    assert heatmaps[0, 2].eq(1).sum() == 2 and heatmaps[0, :2].sum() == 0  # class 3 only
    model = MultiTaskModel(_StubBackbone(), detail_stem=True, heatmap_aux=True)
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert torch.isfinite(losses['loss_heatmap']) and losses['loss_heatmap'] > 0
    sum(losses.values()).backward()
    assert model.heatmap_head.head[0].weight.grad.abs().sum() > 0
    assert model.detail_stem.stem[0].weight.grad.abs().sum() > 0  # the signal reaches the P2 stem
    model.eval()
    with torch.no_grad():
        _, _, losses = model(IMAGES)
    assert losses == {}


def test_cls_detach_keeps_classification_gradients_out_of_the_backbone():
    for detach in (False, True):
        model = MultiTaskModel(_StubBackbone(), input_adapter=True, cls_detach=detach)
        model.train()
        logits, _, _ = model(IMAGES, TARGETS)
        logits.sum().backward()  # the classification path alone
        backbone_grad = model.backbone.patch.weight.grad
        assert model.cls_head[1].weight.grad.abs().sum() > 0  # the classifier itself always trains
        assert (backbone_grad is None or not backbone_grad.any()) == detach


def test_fused_roi_pooling_reads_every_level_and_canonical_scale_reaches_torchvision():
    from torchvision.ops import MultiScaleRoIAlign
    from sarbench.model import FusedRoIAlign
    features = {str(i): torch.zeros(1, 4, 128 // 2 ** i, 128 // 2 ** i) for i in range(4)}
    features['2'][:, :, 4:8, 4:8] = 5.0  # signal only on the stride-16 level
    boxes = [torch.tensor([[64., 64., 128., 128.]])]
    pooled = FusedRoIAlign(['0', '1', '2', '3'], 7, 2)(features, boxes, [(512, 512)])
    assert pooled.shape == (1, 4, 7, 7) and pooled.max() == 5.0  # a 64 px box sees the P4 signal
    single = MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2)(features, boxes, [(512, 512)])
    assert single.max() == 0  # torchvision pools a 64 px box from P2 only (canonical 224)
    model = MultiTaskModel(_StubBackbone(), roi_canonical_scale=56)
    assert model.roi_heads.box_roi_pool.canonical_scale == 56
    fused = MultiTaskModel(_StubBackbone(), roi_pooling='max')
    fused.train()
    _, _, losses = fused(IMAGES, TARGETS)
    assert all(torch.isfinite(v) for v in losses.values())
    with pytest.raises(ValueError):
        MultiTaskModel(_StubBackbone(), roi_pooling='sum')


class _PrefixStubBackbone(_StubBackbone):
    """The stub with DINOv3's `with_prefix` flag (one CLS token, set to ones so it is recognizable)."""

    def forward(self, x, with_prefix=False):
        patches = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([torch.ones(x.shape[0], 1, self.embed_dim), patches], 1)
        return tokens if with_prefix else tokens[:, 1:]


def test_cls_mean_classifier_reads_the_cls_token_and_the_mean_patch_token():
    model = MultiTaskModel(_PrefixStubBackbone(), cls_input='cls_mean')
    assert model.cls_head[1].in_features == 2 * 128
    seen = []
    model.cls_head.register_forward_hook(lambda m, args, out: seen.append(args[0]))
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(IMAGES)
    assert logits.shape == (2, 9) and torch.equal(seen[0][:, :128], torch.ones(2, 128))
    with pytest.raises(ValueError):
        MultiTaskModel(_PrefixStubBackbone(), cls_input='cls_mean', fusion_layers=[1, 2])


def test_roi_sampler_takes_the_configured_batch_and_positive_fraction():
    from torchvision.ops import MultiScaleRoIAlign
    from sarbench.model import MultiTaskModel
    pool = MultiScaleRoIAlign(['0'], 7, 2)
    for head in ('standard', 'cascade'):
        heads = MultiTaskModel._build_roi_heads(pool, 7, 9, head, {'hidden_dim': 64}, 0.0, 0.05, 0.5, 100,
                                                batch_size_per_image=256, positive_fraction=0.5)
        samplers = [heads.fg_bg_sampler, *getattr(heads, 'cascade_samplers', [])]
        for sampler in samplers:
            assert (sampler.batch_size_per_image, sampler.positive_fraction) == (256, 0.5)


def test_alpha_giou_is_giou_at_alpha_one_and_zero_for_a_perfect_box():
    from torchvision.ops import generalized_box_iou
    from sarbench.losses import alpha_giou_loss
    gt = torch.tensor([[10., 10., 20., 20.], [0., 0., 8., 8.], [5., 5., 9., 9.]])
    predicted = torch.tensor([[11., 12., 21., 19.], [0., 0., 8., 8.], [30., 30., 34., 34.]])
    giou = 1 - torch.diag(generalized_box_iou(predicted, gt))
    assert torch.allclose(alpha_giou_loss(predicted, gt, alpha=1.0), giou, atol=1e-5)
    loss = alpha_giou_loss(predicted, gt, alpha=3.0)
    assert loss[1].abs() < 1e-5  # exact box
    assert loss[0] > giou[0]  # a good but imperfect box keeps more loss than GIoU gives it (1 - IoU^3 > 1 - IoU)
    assert 1.0 < loss[2] < 2.0  # disjoint: 1 + penalty^3


@pytest.mark.parametrize('head,params', [('standard', {'hidden_dim': 64}), ('cascade', CASCADE)])
def test_giou_alpha_and_soft_nms_reach_both_heads_train_and_predict(head, params):
    model = MultiTaskModel(_StubBackbone(), giou_weight=1.0, giou_alpha=3.0, soft_nms_sigma=0.5, score_thresh=0.0,
                           detections_per_img=20, head=head, head_params=params)
    assert (model.roi_heads.giou_alpha, model.roi_heads.soft_nms_sigma) == (3.0, 0.5)
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert torch.isfinite(losses['loss_box_reg'])
    losses['loss_box_reg'].backward()
    model.eval()
    with torch.no_grad():
        _, detections, _ = model(IMAGES)
    for d in detections:
        assert len(d['boxes']) <= 20 and len(d['boxes']) == len(d['scores']) == len(d['labels'])
        assert (d['scores'][:-1] >= d['scores'][1:] - 1e-6).all()  # best first


def test_soft_nms_decays_overlaps_instead_of_removing_them_and_leaves_other_classes_alone():
    from sarbench.heads import soft_nms
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 0., 11., 10.], [0., 0., 10., 10.], [50., 50., 60., 60.]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([1, 1, 2, 1])
    keep, kept = soft_nms(boxes, scores, labels, sigma=0.5, min_score=0.001, max_keep=10)
    assert sorted(keep.tolist()) == [0, 1, 2, 3]  # hard NMS at 0.5 would drop box 1 (IoU 0.82)
    decayed = dict(zip(keep.tolist(), kept.tolist()))
    assert decayed[0] == pytest.approx(0.9) and decayed[2] == pytest.approx(0.7)  # class 2 untouched
    assert decayed[3] == pytest.approx(0.6)  # disjoint box untouched
    iou = 90 / 110
    assert decayed[1] == pytest.approx(0.8 * torch.exp(torch.tensor(-iou ** 2 / 0.5)).item(), rel=1e-4)
    keep, _ = soft_nms(boxes, scores, labels, sigma=0.5, min_score=0.5, max_keep=10)
    assert 1 not in keep.tolist()  # pruned once decayed below min_score
    keep, _ = soft_nms(boxes, scores, labels, sigma=0.5, max_keep=2)
    assert keep.tolist() == [0, 2]


def test_soft_nms_sigma_zero_keeps_torchvision_postprocessing():
    from torchvision.models.detection.roi_heads import RoIHeads
    heads = MultiTaskModel(_StubBackbone(), score_thresh=0.0, nms_thresh=0.5).roi_heads
    torch.manual_seed(0)
    proposals = [torch.rand(30, 2) * 50 + torch.tensor([[0., 0.]]).repeat(30, 1)]
    proposals = [torch.cat([proposals[0], proposals[0] + 5 + torch.rand(30, 2) * 20], dim=1)]
    logits, regression = torch.randn(30, 10), torch.randn(30, 40) * 0.1
    ours = heads.postprocess_detections(logits, regression, proposals, [(128, 128)])
    reference = RoIHeads.postprocess_detections(heads, logits, regression, proposals, [(128, 128)])
    assert all(torch.equal(a[0], b[0]) for a, b in zip(ours, reference))  # boxes, scores, labels
    heads.soft_nms_sigma = 0.5  # Soft-NMS keeps at least as many boxes (it decays instead of deleting)
    soft = heads.postprocess_detections(logits, regression, proposals, [(128, 128)])
    assert len(soft[0][0]) >= len(reference[0][0])


def test_roi_aligned_samples_cell_centres_on_both_poolers_and_defaults_to_torchvision():
    from torchvision.ops import MultiScaleRoIAlign
    from sarbench.model import AlignedMultiScaleRoIAlign, FusedRoIAlign
    # feature value = column index; a box over image px [20, 24] is exactly P2 cell 5 (stride 4)
    features = {'0': torch.arange(32.).repeat(32, 1)[None, None]}
    boxes, shapes = [torch.tensor([[20., 20., 24., 24.]])], [(128, 128)]
    fused = FusedRoIAlign(['0'], 1, 1, fuse='max', aligned=True)
    assert fused(features, boxes, shapes).item() == pytest.approx(5.0)
    assert FusedRoIAlign(['0'], 1, 1, fuse='max')(features, boxes, shapes).item() == pytest.approx(5.5)  # old
    features4 = {name: torch.arange(32. / 2 ** i).repeat(int(32 / 2 ** i), 1)[None, None] for i, name in
                 enumerate('0123')}
    level = AlignedMultiScaleRoIAlign(list('0123'), 1, 1)
    assert level(features4, boxes, shapes).item() == pytest.approx(5.0)
    reference = MultiScaleRoIAlign(list('0123'), 1, 1)(features4, boxes, shapes).item()
    assert reference == pytest.approx(5.5)
    for pooling in ('level', 'max'):
        default = MultiTaskModel(_StubBackbone(), roi_pooling=pooling).roi_heads.box_roi_pool
        aligned = MultiTaskModel(_StubBackbone(), roi_pooling=pooling, roi_aligned=True).roi_heads.box_roi_pool
        assert type(default) in (MultiScaleRoIAlign, FusedRoIAlign) and not getattr(default, 'aligned', False)
        assert isinstance(aligned, AlignedMultiScaleRoIAlign) or aligned.aligned


@pytest.mark.parametrize('aligned,centre', [(False, 24.5), (True, 26.0)])
def test_aligned_stem_centres_p2_cells_where_the_vit_path_does(aligned, centre):
    """Impulse test on the stem's two stride-2 convs (ones weights): the mean image coordinate (pixel i covers
    [i, i + 1]) reaching P2 cell 6. The ViT path's cell 6 is centred at (6 + 0.5) x 4 = 26."""
    stem = ConvStem(in_chans=1, channels=8, aligned=aligned)
    convs = [m for m in stem.stem if isinstance(m, nn.Conv2d)][:2]
    hits = []
    with torch.no_grad():
        for conv in convs:
            conv.weight.fill_(1.0)
        for p in range(64):
            x = torch.zeros(1, 1, 64, 64)
            x[..., 30, p] = 1
            y = convs[1](convs[0](x))
            if y[0, :, 7, 6].abs().sum() > 0:
                hits.append(p + 0.5)
    assert sum(hits) / len(hits) == pytest.approx(centre)
    assert stem(torch.randn(1, 1, 64, 64)).shape[-2:] == (16, 16)


def test_size_adaptive_rpn_makes_near_miss_anchors_on_tiny_boxes_positive_and_keeps_large_boxes():
    from torchvision.models.detection._utils import Matcher
    from torchvision.ops import box_iou
    from sarbench.heads import size_adaptive_similarity, size_adaptive_threshold
    gt = torch.tensor([[10., 10., 19., 19.], [100., 100., 500., 500.]])  # 9 px and 400 px
    assert size_adaptive_threshold(gt).tolist() == pytest.approx([0.25, 0.7], abs=1e-3)
    anchors = torch.tensor([[12., 12., 22., 22.],     # IoU ~0.4 with the 9 px box: a near miss
                            [30., 30., 40., 40.],     # nowhere near: background
                            [100., 100., 450., 450.]])  # IoU 0.77 with the 400 px box
    matcher = Matcher(0.7, 0.3, allow_low_quality_matches=False)
    plain = matcher(box_iou(gt, anchors)).tolist()
    adaptive = matcher(size_adaptive_similarity(box_iou)(gt, anchors)).tolist()
    assert plain[0] < 0 and adaptive[0] == 0  # the near miss becomes a positive for the tiny box
    assert plain[1] == adaptive[1] == Matcher.BELOW_LOW_THRESHOLD
    assert plain[2] == adaptive[2] == 1  # large boxes keep the 0.7 bar
    model = MultiTaskModel(_StubBackbone(), rpn_size_adaptive={'floor': 0.25, 'base': 0.2, 'slope': 0.15, 'ref': 12})
    assert model.rpn.box_similarity is not model.roi_heads.box_similarity  # the RoI head keeps plain IoU
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert torch.isfinite(losses['loss_objectness'])


def test_centred_anchors_sit_on_cell_centres_and_train():
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.image_list import ImageList
    from sarbench.heads import CentredAnchorGenerator
    images, features = ImageList(torch.zeros(1, 3, 16, 16), [(16, 16)]), [torch.zeros(1, 1, 4, 4)]
    plain = AnchorGenerator(((8,),), ((1.0,),))(images, features)[0]
    centred = CentredAnchorGenerator(((8,),), ((1.0,),))(images, features)[0]
    assert torch.equal(centred - plain, torch.full_like(plain, 2.0))  # stride 4: half a cell
    centres = (centred[:, :2] + centred[:, 2:]) / 2
    assert centres[:4, 0].tolist() == [2.0, 6.0, 10.0, 14.0]  # (j + 0.5) x 4
    model = MultiTaskModel(_StubBackbone(), rpn_centred_anchors=True)
    assert isinstance(model.rpn.anchor_generator, CentredAnchorGenerator)
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert torch.isfinite(losses['loss_rpn_box_reg'])


def test_context_roi_align_starts_as_the_wrapped_pooler_and_reads_outside_the_box():
    from sarbench.model import ContextRoIAlign, FusedRoIAlign
    pooler = FusedRoIAlign(['0'], (1, 2), 1, fuse='max', aligned=True)
    context = ContextRoIAlign(pooler, scale=3.0, channels=2)
    assert sum(p.numel() for p in context.parameters()) == 2 * 2 * 2 + 2
    features = {'0': torch.arange(32.).repeat(32, 1)[None, None].repeat(1, 2, 1, 1)}  # value = column index (P2)
    boxes, shapes = [torch.tensor([[40., 40., 48., 48.], [0., 0., 8., 8.]])], [(128, 128)]
    assert torch.equal(context(features, boxes, shapes), pooler(features, boxes, shapes))  # identity at init
    enlarged = context.enlarge(boxes, shapes)[0]
    assert enlarged.tolist() == [[32., 32., 56., 56.], [0., 0., 16., 16.]]  # 3x around the centre, clipped
    with torch.no_grad():  # route the context channels only: the two samples spread to the wider window
        context.fuse.weight.zero_()
        context.fuse.weight[:, 2:, 0, 0] = torch.eye(2)
    own = pooler(features, boxes[:1], shapes)[0, 0, 0].tolist()
    wide = context(features, boxes[:1], shapes)[0, 0, 0].tolist()
    assert own == pytest.approx([10., 11.]) and wide == pytest.approx([9., 12.])


def test_roi_context_scale_wraps_either_pooler_and_adds_only_the_fuse_conv():
    from sarbench.model import ContextRoIAlign
    for pooling in ('level', 'max'):
        plain = MultiTaskModel(_StubBackbone(), roi_pooling=pooling)
        model = MultiTaskModel(_StubBackbone(), roi_pooling=pooling, roi_context_scale=3.0)
        assert isinstance(model.roi_heads.box_roi_pool, ContextRoIAlign)
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in plain.parameters())
        assert extra == 512 * 256 + 256
