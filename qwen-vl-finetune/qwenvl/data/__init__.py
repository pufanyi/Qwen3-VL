import json
import os
import re
from functools import lru_cache
from pathlib import Path

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
}


@lru_cache(maxsize=1)
def _external_data_dict():
    """Load additional dataset definitions from a JSON config file if available."""
    config_path = os.getenv("QWENVL_DATA_CONFIG")
    if not config_path:
        repo_root = Path(__file__).resolve().parents[3]
        default_path = repo_root / "scripts" / "data.json"
        if default_path.exists():
            config_path = str(default_path)
    if not config_path:
        return {}

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)

    extra_dict = {}
    for name, cfg in raw_config.items():
        annotation = cfg.get("annotation") or cfg.get("annotation_path")
        if not annotation:
            raise ValueError(
                f"`annotation` field is required for dataset '{name}' in {config_file}"
            )
        data_path = cfg.get("root") or cfg.get("data_path", "")
        extra_cfg = {
            "annotation_path": annotation,
            "data_path": data_path,
        }
        # Keep optional keys to allow downstream components to consume them if needed.
        for optional_key in (
            "repeat_time",
            "length",
            "data_augment",
            "max_dynamic_patch",
        ):
            if optional_key in cfg:
                extra_cfg[optional_key] = cfg[optional_key]
        extra_dict[name] = extra_cfg
    return extra_dict


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    available_data = {**data_dict, **_external_data_dict()}
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in available_data:
            config = available_data[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
