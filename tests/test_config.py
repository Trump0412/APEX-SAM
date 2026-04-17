from apex_sam.config import ApexConfig


def test_config_defaults():
    cfg = ApexConfig()
    assert cfg.prompt_mode == 'voronoi'
    assert cfg.enable_dino_freq_fusion is True
    assert not hasattr(cfg, 'disable_freq_style')
    assert cfg.force_input_size == 256
    assert cfg.eval_protocol == "slice_mean"
