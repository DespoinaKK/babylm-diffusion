from __future__ import annotations

import json
import pathlib
import copy
from transformers.configuration_utils import PretrainedConfig


class ModelConfig(PretrainedConfig):

    def __init__(self: ModelConfig, config_file: pathlib.Path | str | None = None, **kwargs):
        """
        """
        super().__init__(**kwargs)
        if config_file is None:
            self.attention_probs_dropout_prob: float = 0.1
            self.hidden_dropout_prob = 0.1
            self.hidden_size = 768
            self.intermediate_size = 2560
            self.max_position_embeddings = 512
            self.position_bucket_size = 32
            self.num_attention_heads = 12
            self.num_layers = 12
            self.vocab_size = 16384
            self.layer_norm_eps = 1.0e-7
            self.cond_dim = 128
        else:
            # if type(config_file) == str:
            config_file = pathlib.Path(config_file)

            config = json.load(config_file.open("r"))

            for key, value in config.items():
                setattr(self, key, value)

    def __repr__(self) -> str:
        return str(self.to_json_string())

    def to_dict(self) -> dict:
        """Serializes this instance to a Python dictionary."""
        output: dict

        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: pathlib.Path | str) -> None:
        """Save this instance to a json file."""
        if isinstance(json_file_path, str):
            json_file_path: pathlib.Path = pathlib.Path(json_file_path)
        with json_file_path.open("w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())
