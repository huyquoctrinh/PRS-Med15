from prs_med15.config import DataConfig, load_config


def test_typed_config_and_override():
    cfg = load_config("configs/base.yaml", ["data.image_size=64", "runtime.seed=7"])
    assert isinstance(cfg.data, DataConfig)
    assert cfg.data.image_size == 64 and cfg.runtime.seed == 7
    model = load_config("configs/model/shared_slice.yaml")
    assert model.model["encoder"]["image_size"] == 512
    stage = load_config("configs/train/stage_a.yaml")
    assert stage.train["stage"] == "a"
