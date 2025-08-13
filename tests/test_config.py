import yaml
import argparse
from pathlib import Path
from pyv_npu.config import SimConfig

def test_config_yaml_loading(tmp_path: Path):
    """Tests that config is loaded correctly from a YAML file."""
    yaml_content = {
        'level': 'L2',
        'te': 16,
        'mode': 'tight'
    }
    yaml_file = tmp_path / "test.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(yaml_file))
    parser.add_argument("--model", default="test.onnx")
    args = parser.parse_args([])

    config = SimConfig.from_args(args)

    assert config.level == 'L2'
    assert config.te == 16
    assert config.mode == 'tight'

def test_config_cli_override(tmp_path: Path):
    """Tests that CLI arguments override YAML settings."""
    yaml_content = {
        'level': 'L2',
        'te': 16,
        'mode': 'tight'
    }
    yaml_file = tmp_path / "test.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)

    parser = argparse.ArgumentParser()
    # Simulate providing CLI args
    parser.add_argument("--config", default=str(yaml_file))
    parser.add_argument("--model", default="test.onnx")
    parser.add_argument("--level", default="L3") # Override
    parser.add_argument("--te", type=int, default=8) # Override
    
    # In a real scenario, we'd parse from sys.argv
    # Here, we create a Namespace object to simulate it
    args = argparse.Namespace(config=str(yaml_file), model="test.onnx", level="L3", te=8)

    config = SimConfig.from_args(args)

    assert config.level == 'L3' # Overridden value
    assert config.te == 8      # Overridden value
    assert config.mode == 'tight' # Value from YAML
