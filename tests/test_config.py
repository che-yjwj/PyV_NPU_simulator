import yaml
import argparse
from pathlib import Path
from pyv_npu.config import SimConfig

def test_config_yaml_loading(tmp_path: Path):
    """Tests that config is loaded correctly from a YAML file."""
    yaml_content = {
        'sim_level': 'CA_HYBRID',
        'tc': 16,
        'mode': 'tight'
    }
    yaml_file = tmp_path / "test.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)

    # Simulate args parsed from CLI, where only config and model are provided
    args = argparse.Namespace(config=str(yaml_file), model="test.onnx", sim_level=None, tc=None)

    config = SimConfig.from_args(args)

    assert config.sim_level == 'CA_HYBRID'
    assert config.tc == 16
    assert config.mode == 'tight'

def test_config_cli_override(tmp_path: Path):
    """Tests that CLI arguments override YAML settings."""
    yaml_content = {
        'sim_level': 'CA_HYBRID',
        'tc': 16,
        'mode': 'tight'
    }
    yaml_file = tmp_path / "test.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
    
    # Simulate args parsed from CLI, with values overriding the YAML
    args = argparse.Namespace(
        config=str(yaml_file), 
        model="test.onnx", 
        sim_level="CA_FULL", # Override
        tc=8                 # Override
    )

    config = SimConfig.from_args(args)

    assert config.sim_level == 'CA_FULL' # Overridden value
    assert config.tc == 8                # Overridden value
    assert config.mode == 'tight'        # Value from YAML


def test_config_cycles_conversion():
    """Tests the conversion from seconds to clock cycles."""
    # given
    config = SimConfig()
    config.clock_ghz = 1.2

    # when
    one_second_cycles = config.cycles(1.0)
    half_second_cycles = config.cycles(0.5)
    zero_cycles = config.cycles(0)

    # then
    assert one_second_cycles == 1_200_000_000
    assert half_second_cycles == 600_000_000
    assert zero_cycles == 0
