"""A compact Deformable-DETR detection head (Zhu et al., "Deformable DETR", ICLR 2021).

It consumes the multi-scale feature maps built from the ViT backbone by SimpleFeaturePyramid and predicts a
fixed set of boxes directly (no anchors, no NMS), which suits small SAR objects because the deformable
attention samples sparsely around learned reference points at every pyramid level (including the stride-4 P2).

Training uses the standard DETR recipe: Hungarian matching + cross-entropy (with a no-object class), L1 and
GIoU losses. Evaluation returns torchvision-style per-image {'boxes' (XYXY, input pixels), 'labels', 'scores'}
so the rest of the pipeline (metrics, submission) is unchanged.

Parameter budget with the defaults (hidden 256, 6 encoder + 6 decoder layers, 4 levels, 300 queries, 8 heads,
4 points): about 11 M, independent of the ViT width (features are projected to 256).
"""
import math
from copy import deepcopy

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.ops import box_convert, generalized_box_iou


def _get_clones(module, n):
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def _sine_pos(h, w, d_model, device):
    """Fixed 2-D sine positional encoding for one feature level: (1, h * w, d_model), no parameters."""
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    num = d_model // 2
    dim = 1000 ** (2 * (torch.arange(num, device=device, dtype=torch.float32) // 2) / num)
    pos_x = x.flatten()[:, None].float() / dim
    pos_y = y.flatten()[:, None].float() / dim
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    return torch.cat((pos_y, pos_x), dim=1).unsqueeze(0)


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention: each query samples `n_points` per head and feature level."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_points, value, spatial_shapes, level_start_index):
        # query (N, Lq, C); value (N, Lin, C); spatial_shapes (L, 2) as (h, w); reference_points (N, Lq, L, 2)
        n, len_q, _ = query.shape
        n, len_in, _ = value.shape
        value = self.value_proj(value).view(n, len_in, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(n, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(query).view(n, len_q, self.n_heads, self.n_levels * self.n_points)
        weights = F.softmax(weights, -1).view(n, len_q, self.n_heads, self.n_levels, self.n_points)
        normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], -1).to(query.dtype)
        locations = reference_points[:, :, None, :, None, :] \
            + offsets / normalizer[None, None, None, :, None, :]
        value_list = value.split([h * w for h, w in spatial_shapes.tolist()], dim=1)
        sampling_grids = 2 * locations - 1
        # Accumulate the levels one at a time: a single stacked tensor over all levels is many GiB at P2
        # resolution, whereas each level's sampled tensor is `n_levels` times smaller.
        weights = weights.transpose(1, 2)  # (n, heads, len_q, n_levels, n_points)
        output = None
        for level, (h, w) in enumerate(spatial_shapes.tolist()):
            value_l = value_list[level].flatten(2).transpose(1, 2).reshape(
                n * self.n_heads, self.d_model // self.n_heads, h, w)
            grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            sampled_l = F.grid_sample(value_l, grid_l, mode='bilinear', padding_mode='zeros',
                                      align_corners=False)  # (n*heads, C/heads, len_q, n_points)
            weight_l = weights[:, :, :, level].reshape(n * self.n_heads, 1, len_q, self.n_points)
            term = (sampled_l * weight_l).sum(-1)  # (n*heads, C/heads, len_q)
            output = term if output is None else output + term
        output = output.view(n, self.d_model, len_q).transpose(1, 2).contiguous()
        return self.output_proj(output)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        src = self.norm1(src + self.dropout(self.self_attn(src + pos, reference_points, src,
                                                            spatial_shapes, level_start_index)))
        src = self.norm2(src + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(src))))))
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers, grad_checkpointing=False):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.grad_checkpointing = grad_checkpointing

    def forward(self, src, spatial_shapes, level_start_index, pos):
        reference_points = _reference_points(spatial_shapes, src.device).unsqueeze(0).repeat(src.shape[0], 1, 1, 1)
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                # Recompute the layer in backward instead of storing its (large) attention activations.
                src = checkpoint(layer, src, pos, reference_points, spatial_shapes, level_start_index,
                                 use_reentrant=False)
            else:
                src = layer(src, pos, reference_points, spatial_shapes, level_start_index)
        return src


def _reference_points(spatial_shapes, device):
    """Normalized [0, 1] (x, y) grid for every level, concatenated: (sum(h*w), n_levels, 2)."""
    points = []
    for h, w in spatial_shapes.tolist():
        ys = (torch.arange(h, device=device) + 0.5) / h
        xs = (torch.arange(w, device=device) + 0.5) / w
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        points.append(torch.stack([grid_x.flatten(), grid_y.flatten()], -1))
    points = torch.cat(points, 0)  # (sumHW, 2)
    return points[:, None, :].repeat(1, spatial_shapes.shape[0], 1)  # (sumHW, n_levels, 2)


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, query_pos, reference_points, src, spatial_shapes, level_start_index):
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.dropout(self.self_attn(q, k, tgt)[0]))
        tgt = self.norm2(tgt + self.dropout(self.cross_attn(tgt + query_pos, reference_points, src,
                                                            spatial_shapes, level_start_index)))
        tgt = self.norm3(tgt + self.dropout(self.linear2(self.dropout(self.activation(self.linear1(tgt))))))
        return tgt


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class HungarianMatcher(nn.Module):
    """One-to-one assignment between predictions and targets, by class + L1 + GIoU cost."""

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # (B*Q, C+1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # (B*Q, 4), cxcywh in [0, 1]
        indices = []
        for b in range(bs):
            tgt_ids = targets[b]['labels'] - 1  # 0-based object classes
            tgt_bbox = targets[b]['boxes_norm']
            if len(tgt_ids) == 0:  # no objects: nothing to match (keeps linear_sum_assignment happy)
                indices.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue
            cost_class = -out_prob[b * num_queries:(b + 1) * num_queries][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b * num_queries:(b + 1) * num_queries], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_convert(out_bbox[b * num_queries:(b + 1) * num_queries],
                                                         'cxcywh', 'xyxy'),
                                             box_convert(tgt_bbox, 'cxcywh', 'xyxy'))
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            row, col = linear_sum_assignment(cost.cpu())
            indices.append((torch.as_tensor(row, device=out_prob.device),
                            torch.as_tensor(col, device=out_prob.device)))
        return indices


class DeformableDetrLoss(nn.Module):
    """DETR set loss: CE over classes (with a no-object class) + L1 and GIoU box losses."""

    def __init__(self, num_classes=9, matcher=None, eos_coef=0.1, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0,
                 aux_loss=True):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.aux_loss = aux_loss
        weight = torch.ones(num_classes + 1)
        weight[num_classes] = eos_coef  # down-weight the no-object class
        self.register_buffer('class_weight', weight)

    @staticmethod
    def _permutation(indices):
        nonempty = [(src, tgt) for src, tgt in indices if len(src)]
        if not nonempty:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        batch = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(nonempty)])
        src = torch.cat([src for src, _ in nonempty])
        return batch, src

    def _class_targets(self, logits, indices, targets):
        classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device)
        for b, (_, tgt) in enumerate(indices):
            if len(tgt):
                classes[b, indices[b][0]] = targets[b]['labels'][tgt] - 1  # 0-based object class
        return classes

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        batch, src = self._permutation(indices)
        losses = {'loss_classifier': F.cross_entropy(
            outputs['pred_logits'].transpose(1, 2), self._class_targets(outputs['pred_logits'], indices, targets),
            self.class_weight.to(outputs['pred_logits'].dtype))}

        num_boxes = max(1, sum(len(t['labels']) for t in targets))
        if len(batch):  # images do have objects (always true for this dataset, guarded for safety)
            pred = outputs['pred_boxes'][batch, src]
            gt = torch.cat([t['boxes_norm'][tgt] for t, (_, tgt) in zip(targets, indices) if len(tgt)])
            losses['loss_bbox'] = F.l1_loss(pred, gt, reduction='none').sum() / num_boxes
            losses['loss_giou'] = (1 - torch.diag(generalized_box_iou(
                box_convert(pred, 'cxcywh', 'xyxy'), box_convert(gt, 'cxcywh', 'xyxy')))).sum() / num_boxes
        else:
            losses['loss_bbox'] = outputs['pred_boxes'].sum() * 0.
            losses['loss_giou'] = outputs['pred_boxes'].sum() * 0.

        if self.aux_loss and 'aux_outputs' in outputs:
            losses['loss_aux'] = 0.
            for aux in outputs['aux_outputs']:
                aux_indices = self.matcher(aux, targets)
                losses['loss_aux'] = losses['loss_aux'] + F.cross_entropy(
                    aux['pred_logits'].transpose(1, 2), self._class_targets(aux['pred_logits'], aux_indices, targets),
                    self.class_weight.to(aux['pred_logits'].dtype))
        return losses


class DeformableDetrHead(nn.Module):
    """Multi-scale feature maps -> (per-image detections on eval, losses on train)."""

    def __init__(self, in_channels=256, num_classes=9, num_queries=300, level_names=('0', '1', '2', '3'),
                 d_model=256, n_heads=8, n_points=4, enc_layers=6, dec_layers=6, dim_feedforward=1024,
                 dropout=0.1, score_thresh=0.05, detections_per_img=100, aux_loss=True,
                 cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, eos_coef=0.1, grad_checkpointing=False):
        super().__init__()
        self.num_classes, self.num_queries = num_classes, num_queries
        self.level_names = list(level_names)
        self.n_levels = len(self.level_names)
        self.score_thresh, self.detections_per_img, self.aux_loss = score_thresh, detections_per_img, aux_loss
        self.input_proj = nn.ModuleList(nn.Conv2d(in_channels, d_model, 1) for _ in range(self.n_levels))
        self.level_embed = nn.Parameter(torch.zeros(self.n_levels, d_model))
        self.encoder = DeformableTransformerEncoder(
            DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, self.n_levels, n_heads, n_points),
            enc_layers, grad_checkpointing)
        self.decoder_layers = _get_clones(
            DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, self.n_levels, n_heads, n_points),
            dec_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.reference_points = nn.Linear(d_model, 2)
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.criterion = DeformableDetrLoss(num_classes, matcher=HungarianMatcher(cost_class, cost_bbox, cost_giou),
                                            eos_coef=eos_coef, aux_loss=aux_loss)
        nn.init.zeros_(self.class_embed.bias)
        nn.init.constant_(self.class_embed.bias, -math.log(1 / (num_classes + 1)))
        nn.init.constant_(self.reference_points.weight, 0.)
        nn.init.constant_(self.reference_points.bias, 0.)

    def _features(self, features):
        """Selected neck levels -> flat src, spatial_shapes, level_start_index, sine pos."""
        src, pos, shapes = [], [], []
        for name, conv in zip(self.level_names, self.input_proj):
            feat = conv(features[name])
            n, _, h, w = feat.shape
            shapes.append((h, w))
            src.append(feat.flatten(2).transpose(1, 2))
            pos.append(_sine_pos(h, w, feat.shape[1], feat.device).to(feat.dtype))
        spatial_shapes = torch.as_tensor(shapes, device=src[0].device, dtype=torch.long)
        lengths = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        level_start_index = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)[:-1]])
        src = torch.cat([part + self.level_embed[level] for level, part in enumerate(src)], 1)
        return src, torch.cat(pos, 1), spatial_shapes, level_start_index

    def forward(self, features, targets=None, image_size=None):
        src, pos, spatial_shapes, level_start_index = self._features(features)
        memory = self.encoder(src, spatial_shapes, level_start_index, pos)

        n = memory.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, n, 1)  # (Q, N, C)
        query_embed = query_embed.transpose(0, 1)  # (N, Q, C)
        target = torch.zeros_like(query_embed)
        reference_points = self.reference_points(query_embed).sigmoid()  # (N, Q, 2)

        intermediate_logits, intermediate_boxes = [], []
        for layer in self.decoder_layers:
            target = layer(target, query_embed, reference_points[:, :, None, :].repeat(1, 1, self.n_levels, 1),
                           memory, spatial_shapes, level_start_index)
            delta = self.bbox_embed(target)  # (N, Q, 4) raw cxcywh deltas
            center = delta[..., :2] + inverse_sigmoid(reference_points)  # refine the reference (cx, cy)
            boxes = torch.cat([center, delta[..., 2:]], -1).sigmoid()  # normalized cxcywh
            reference_points = boxes[..., :2].detach()  # next layer attends around the refined center
            intermediate_logits.append(self.class_embed(target))
            intermediate_boxes.append(boxes)

        outputs = {'pred_logits': intermediate_logits[-1], 'pred_boxes': intermediate_boxes[-1]}
        if self.aux_loss:
            outputs['aux_outputs'] = [{'pred_logits': logits, 'pred_boxes': boxes}
                                      for logits, boxes in zip(intermediate_logits[:-1], intermediate_boxes[:-1])]

        if self.training and targets is not None:
            return [], self.criterion(outputs, self._prepare_targets(targets, image_size))
        return self._detections(outputs['pred_logits'], outputs['pred_boxes'], image_size), {}

    @staticmethod
    def _prepare_targets(targets, image_size):
        """Normalized cxcywh targets; boxes are XYXY pixels on the input canvas (all images 512 x 512)."""
        height, width = image_size
        prepared = []
        for target in targets:
            prepared.append({'labels': target['labels'],
                             'boxes_norm': box_convert(target['boxes'], 'xyxy', 'cxcywh')
                             / target['boxes'].new_tensor([width, height, width, height])})
        return prepared

    def _detections(self, logits, boxes, image_size):
        """threshold + top-k over the object classes -> torchvision-style detections in input pixels."""
        height, width = image_size
        probs = logits.softmax(-1)[..., :self.num_classes]  # drop the no-object class
        scores, labels = probs.max(-1)
        scale = torch.tensor([width, height, width, height], device=boxes.device, dtype=boxes.dtype)
        detections = []
        for i in range(logits.shape[0]):
            keep = scores[i] > self.score_thresh
            b, c, s = boxes[i][keep], labels[i][keep] + 1, scores[i][keep]  # labels 1..num_classes
            if len(s) > self.detections_per_img:
                s, top = s.topk(self.detections_per_img)
                b, c = b[top], c[top]
            detections.append({'boxes': box_convert(b, 'cxcywh', 'xyxy') * scale, 'labels': c, 'scores': s})
        return detections
