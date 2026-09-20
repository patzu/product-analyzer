from vision_sorter.config import Config


def test_paths_and_initialization(tmp_path):
    config = Config(root=tmp_path, report_dir="output")
    config.initialize()
    assert config.report_dir == tmp_path / "output"
    assert (config.data_dir / "raw").is_dir()
