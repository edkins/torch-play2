import jsonschema
import pickle
import yaml

config_schema = {
    "type": "object",
    "properties": {
        "version": {"const":"0.1"},
        "metadata": {
            "type": "object",
            "properties": {
                "name": {"type":"string"},
                "annotations": {
                    "type": "object",
                    "additionalProperties": {"type":"string"}
                }
            },
            "required": ["name"]
        },
        "data": {
            "type": "object",
            "properties": {
                "src": {"type": "string"}
            },
            "required": ["src"]
        },
        "layers": {
            "type": "array",
            "item": {
                "type": "object",
                "properties": {
                    "type": {"type":"string"},
                    "size": {
                        "type":["integer","string"],
                        "regex": "^[0-9]+(x[0-9]+)*$"
                    },
                    "kernel_size": {"type":"integer"},
                    "padding": {"type":"integer"}
                },
                "required": ["type"]
            }
        },
        "training": {
            "properties": {
                "batch_size": {"type": "integer"},
                "device": {"type": "string"},
                "epochs": {"type": "integer"},
                "optimizer": {
                    "type": "object",
                    "properties": {
                        "type": {"type":"string"}
                    }
                },
                "loss": {
                    "type": "object",
                    "properties": {
                        "type": {"type":"string"}
                    }
                }
            },
            "required": ["epochs","batch_size"]
        }
    },
    "required": ["metadata","data","layers","training","version"]
}

def get_config(filename: str = 'config.yaml') -> tuple[str,dict]:
    with open(filename) as f:
        text = f.read()
        config = yaml.safe_load(text)
        jsonschema.validate(instance=config, schema=config_schema)
        return text, config


def get_results(filename: str) -> dict:
    with open(filename) as f:
        text = f.read()
        return yaml.safe_load(text)


def get_state_dict(filename: str) -> dict:
    with open(filename, 'rb') as f:
        return pickle.load(f)
