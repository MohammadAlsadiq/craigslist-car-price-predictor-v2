"""Shared configuration and logging helpers."""
import logging
from pathlib import Path
import yaml

def setup_logging():
    """Configure console logging for all modules."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

def load_config(path='config.yaml'):
    """Load YAML configuration or raise a helpful error."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f'Missing configuration: {file}')
    try:
        with file.open(encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid YAML in {file}') from exc
