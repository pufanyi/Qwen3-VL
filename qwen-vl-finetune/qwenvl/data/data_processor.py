import json
import random
import logging
import re
import time
import itertools
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Any, Set
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3
from utils.storage_clients import PatternAOSSClient

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


_REMOTE_SCHEMES = {"http", "https", "s3"}
_DEFAULT_CACHE_SUBDIR = "qwenvl_media_cache"


def _is_remote_path(path: str) -> bool:
    if not path:
        return False
    parsed = urlparse(path)
    return parsed.scheme in _REMOTE_SCHEMES


def _ensure_cache_dir(cache_dir: Optional[str]) -> Path:
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
    else:
        cache_path = Path(tempfile.gettempdir()) / _DEFAULT_CACHE_SUBDIR
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _download_remote_file(uri: str, storage_client, cache_dir: Optional[str]) -> str:
    if storage_client is None or not uri.startswith("s3://"):
        return uri

    cache_dir_path = _ensure_cache_dir(cache_dir)
    file_ext = Path(uri).suffix
    hashed_name = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    target_path = cache_dir_path / f"{hashed_name}{file_ext}"

    if not target_path.exists():
        try:
            data = storage_client.get(uri)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch remote media '{uri}': {exc}") from exc

        if data is None:
            raise FileNotFoundError(f"Remote media not found or empty: {uri}")

        if isinstance(data, memoryview):
            data_bytes = data.tobytes()
        elif isinstance(data, (bytes, bytearray)):
            data_bytes = bytes(data)
        elif hasattr(data, "read"):
            data_bytes = data.read()
        elif isinstance(data, str):
            # Treat textual response as UTF-8 bytes to avoid crashes; remote assets should not be text.
            data_bytes = data.encode("utf-8")
        else:
            raise TypeError(
                f"Unsupported object type '{type(data)!r}' returned for remote media '{uri}'"
            )

        if data_bytes is None:
            raise RuntimeError(f"Remote media '{uri}' returned no data.")

        target_path.write_bytes(data_bytes)

    return str(target_path)


def _make_abs_paths(base: str, files: str, *, storage_client=None, cache_dir: Optional[str] = None) -> str:
    if not files:
        return ""

    if _is_remote_path(files):
        return _download_remote_file(files, storage_client, cache_dir)

    if _is_remote_path(base):
        base = base.rstrip("/")
        combined = f"{base}/{files.lstrip('/')}"
        return _download_remote_file(combined, storage_client, cache_dir)

    base_path = Path(base).expanduser()
    file_path = Path(files).expanduser()

    if file_path.is_absolute():
        return str(file_path)

    resolved = (base_path / file_path).resolve()
    if storage_client and str(resolved).startswith("s3://"):
        return _download_remote_file(str(resolved), storage_client, cache_dir)
    return str(resolved)


def _parse_aoss_rules(raw_rules) -> Optional[List[Tuple[str, str]]]:
    if not raw_rules:
        return None

    if isinstance(raw_rules, str):
        text = raw_rules.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            rules: List[Tuple[str, str]] = []
            for chunk in text.split(";;"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "::" in chunk:
                    pattern, conf_path = chunk.split("::", 1)
                elif "=" in chunk:
                    pattern, conf_path = chunk.split("=", 1)
                else:
                    raise ValueError(
                        f"Invalid AOSS rule entry '{chunk}'. Expected 'pattern::conf_path'."
                    )
                rules.append((pattern.strip(), conf_path.strip()))
            return rules or None
        else:
            rules: List[Tuple[str, str]] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        pattern = item.get("pattern")
                        conf_path = item.get("conf_path")
                    elif (
                        isinstance(item, (list, tuple))
                        and len(item) == 2
                    ):
                        pattern, conf_path = item
                    else:
                        continue
                    if pattern and conf_path:
                        rules.append((str(pattern), str(conf_path)))
                return rules or None
            if isinstance(parsed, dict):
                pattern = parsed.get("pattern")
                conf_path = parsed.get("conf_path")
                if pattern and conf_path:
                    return [(str(pattern), str(conf_path))]
            raise ValueError(
                "aoss_conf_rules JSON must be a list of objects or an object with 'pattern' and 'conf_path'."
            )

    if isinstance(raw_rules, Sequence):
        rules: List[Tuple[str, str]] = []
        for item in raw_rules:
            if isinstance(item, dict):
                pattern = item.get("pattern")
                conf_path = item.get("conf_path")
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pattern, conf_path = item
            else:
                continue
            if pattern and conf_path:
                rules.append((str(pattern), str(conf_path)))
        return rules or None

    return None


def _sample_contains_remote_media(sample: Dict[str, Any]) -> bool:
    def _check(value) -> bool:
        if isinstance(value, str):
            return _is_remote_path(value)
        if isinstance(value, list):
            return any(_check(v) for v in value)
        if isinstance(value, dict):
            return any(_check(v) for v in value.values())
        return False

    if not isinstance(sample, dict):
        return False

    if _check(sample.get("data_path")):
        return True
    if _check(sample.get("image")) or _check(sample.get("images")):
        return True
    if _check(sample.get("video")) or _check(sample.get("videos")):
        return True

    return False


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)

    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(
            f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}"
        )
        rank0_print(
            f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(
                f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(
            f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
        )
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _build_messages(
    item: Dict[str, Any],
    base_path: str,
    *,
    storage_client=None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = []
    for img in images:
        try:
            resolved = _make_abs_paths(
                base_path, img, storage_client=storage_client, cache_dir=cache_dir
            )
            image_pool.append({"type": "image", "image": resolved})
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Image asset missing for '{img}' (base '{base_path}'): {exc}"
            ) from exc

    video_pool = []
    skipped_videos: List[Tuple[str, Exception]] = []
    for vid in videos:
        try:
            resolved = _make_abs_paths(
                base_path, vid, storage_client=storage_client, cache_dir=cache_dir
            )
            video_pool.append({"type": "video", "video": resolved})
        except FileNotFoundError as exc:
            skipped_videos.append((vid, exc))

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        if skipped_videos:
                            missing_list = ", ".join(v for v, _ in skipped_videos)
                            detail = f" Missing video files: {missing_list}."
                        else:
                            detail = ""
                        raise FileNotFoundError(
                            "Number of <video> placeholders exceeds the number of provided videos." + detail
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    human_text = "".join(
        conv["value"] for conv in item["conversations"] if conv["from"] == "human"
    )
    has_video_placeholder = "<video>" in human_text

    if video_pool and has_video_placeholder:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    if skipped_videos and has_video_placeholder:
        missing_list = ", ".join(f"{vid} ({err})" for vid, err in skipped_videos)
        raise FileNotFoundError(
            f"One or more referenced videos could not be fetched: {missing_list}"
        )


    return messages


def preprocess_qwen_visual(
    sources,
    processor,
    *,
    storage_client=None,
    cache_dir: Optional[str] = None,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = source.get("data_path", "")
    messages = _build_messages(
        source,
        base_path,
        storage_client=storage_client,
        cache_dir=cache_dir,
    )

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict
        self._invalid_sample_indices: Set[int] = set()

        raw_conf_path = getattr(self.data_args, "aoss_conf_path", None) or os.getenv(
            "AOSS_CONF_PATH"
        )
        raw_rules = getattr(self.data_args, "aoss_conf_rules", None) or os.getenv(
            "AOSS_CONF_RULES"
        )
        pattern_rules = _parse_aoss_rules(raw_rules)
        self.media_cache_dir = getattr(self.data_args, "media_cache_dir", None) or os.getenv(
            "QWENVL_MEDIA_CACHE"
        )

        remote_required = any(
            _sample_contains_remote_media(sample) for sample in self.list_data_dict
        )

        self.storage_client = None
        should_init_storage = remote_required or raw_conf_path or pattern_rules
        if should_init_storage:
            try:
                self.storage_client = PatternAOSSClient(raw_conf_path, pattern_rules)
            except ImportError as exc:
                if remote_required:
                    raise ImportError(
                        "AOSS client requested for remote media but aoss-client is not installed."
                    ) from exc
                rank0_print("AOSS client not installed; continuing without remote storage support.")
            except Exception as exc:
                if remote_required:
                    raise RuntimeError(
                        f"Failed to initialize AOSS storage client: {exc}"
                    ) from exc
                rank0_print(f"Warning: failed to initialize AOSS storage client ({exc}); continuing without it.")

        if remote_required and self.storage_client is None:
            raise RuntimeError(
                "Detected remote media (e.g., s3 URIs) but no AOSS storage client is configured. "
                "Please provide --aoss_conf_path/--aoss_conf_rules or set AOSS_CONF_PATH/AOSS_CONF_RULES."
            )

        if self.storage_client:
            rank0_print(
                f"AOSS storage client ready. Media cache directory: "
                f"{self.media_cache_dir or '<system temp>'}"
            )

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def _fetch_with_retries(self, index: int) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        last_exception: Optional[Exception] = None

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[index]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except FileNotFoundError:
                raise
            except Exception as e:
                last_exception = e
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {index}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        next_index = min(index + 1, len(self.list_data_dict) - 1)
        if next_index != index:
            for attempt_idx in range(num_base_retries):
                try:
                    sources = self.list_data_dict[next_index]
                    if isinstance(sources, dict):
                        sources = [sources]

                    sample = self.item_fn(sources)
                    return sample
                except FileNotFoundError:
                    raise
                except Exception as e:
                    last_exception = e
                    # no need to sleep
                    print(
                        f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                        e,
                    )

        # Final attempt on the original sample; propagate any exception.
        sources = self.list_data_dict[index]
        if isinstance(sources, dict):
            sources = [sources]
        try:
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        dataset_size = len(self.list_data_dict)
        if dataset_size == 0:
            raise IndexError("Dataset is empty.")

        idx = i % dataset_size
        checked: Set[int] = set()
        last_missing_error: Optional[FileNotFoundError] = None

        while len(checked) < dataset_size:
            if idx in checked or idx in self._invalid_sample_indices:
                checked.add(idx)
                idx = (idx + 1) % dataset_size
                continue

            try:
                return self._fetch_with_retries(idx)
            except FileNotFoundError as err:
                import traceback
                traceback.print_exc()
                self._invalid_sample_indices.add(idx)
                checked.add(idx)
                last_missing_error = err
                print(f"data_dict: {self.list_data_dict[idx]}")
                print(f"[Skip] Missing remote media for sample {idx}: {err}")
                idx = (idx + 1) % dataset_size
                continue

        if last_missing_error:
            raise RuntimeError(
                "Exhausted dataset because all candidate samples are missing remote media."
            ) from last_missing_error

        raise RuntimeError(
            "Unable to fetch dataset sample after exhausting retry attempts."
        )

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            storage_client=self.storage_client,
            cache_dir=self.media_cache_dir,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )

        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            if isinstance(source, dict):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        if isinstance(sources, list):
            data_list = []
            new_data_dict = {}
            for source in sources:
                if isinstance(source, dict):
                    source = [source]
                assert (
                    len(source) == 1
                ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
                data_list.append(self._get_item(source))

            input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
            labels = torch.cat([d["labels"] for d in data_list], dim=1)
            position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
            attention_mask = [
                d["attention_mask"][0] for d in data_list if "attention_mask" in d
            ]
            new_data_dict = {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "attention_mask": attention_mask if attention_mask else None,
            }

            if any("pixel_values" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values": torch.cat(
                            [
                                d["pixel_values"]
                                for d in data_list
                                if "pixel_values" in d
                            ],
                            dim=0,
                        ),
                        "image_grid_thw": torch.cat(
                            [
                                d["image_grid_thw"]
                                for d in data_list
                                if "image_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )

            if any("pixel_values_videos" in d for d in data_list):
                new_data_dict.update(
                    {
                        "pixel_values_videos": torch.cat(
                            [
                                d["pixel_values_videos"]
                                for d in data_list
                                if "pixel_values_videos" in d
                            ],
                            dim=0,
                        ),
                        "video_grid_thw": torch.cat(
                            [
                                d["video_grid_thw"]
                                for d in data_list
                                if "video_grid_thw" in d
                            ],
                            dim=0,
                        ),
                    }
                )
            return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
