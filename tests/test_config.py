import pytest
import yaml

from sarbench.config import load_config
from train import ROOT, parse_args

FINAL = ROOT / 'configs' / 'final.yaml'


def test_final_config_is_the_d16_recipe():
    config = load_config(FINAL)
    assert config['train']['epochs'] == 250 and config['data']['val_fraction'] == 0
    assert config['loss'] == {'classification': 0.1, 'loss_box_reg': 2.0, 'loss_rpn_box_reg': 1.5}
    assert (config['model']['score_thresh'], config['model']['nms_thresh']) == (1e-5, 0.85)


@pytest.mark.parametrize('edit', [lambda c: c['train'].pop('lr'), lambda c: c['model'].update(head='cascade'),
                                  lambda c: c.pop('loss')])
def test_missing_or_unknown_keys_are_rejected(tmp_path, edit):
    config = yaml.safe_load(FINAL.read_text())
    edit(config)
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_cli_overrides_only_what_is_given():
    _, config = parse_args(['--data', 'x', '--epochs', '1', '--val-fraction', '0.1'])
    assert config['train']['epochs'] == 1 and config['data']['val_fraction'] == 0.1
    assert config['train']['batch_size'] == load_config(FINAL)['train']['batch_size']
