from pathlib import Path

from confz import BaseConfig, FileSource

CONFIG_DIR = Path(__file__).parent.resolve()


class Config(BaseConfig):
    data_paths: dict
    model_paths: dict

    CONFIG_SOURCES = FileSource(file=CONFIG_DIR / "params.yml")

