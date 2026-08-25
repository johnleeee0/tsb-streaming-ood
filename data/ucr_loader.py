from __future__ import annotations

import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from sklearn.model_selection import train_test_split

from .registry import register


@dataclass
class UCRUEAConfig:
    data_dir: str
    dataset_name: str
    archive: str = "UCR"
    base_url: Optional[str] = None
    files_base_url: Optional[Union[str, List[str]]] = None
    zenodo_record_id: Optional[str] = None
    allow_full_archive_download: bool = True
    val_ratio: float = 0.2
    seed: int = 42
    normalize: str = "per_series"
    pad_value: float = 0.0
    augmentations: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None


class TimeSeriesDataset:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        augmentations: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
    ) -> None:
        self.x = x
        self.y = y
        self.augmentations = augmentations or []

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self.x[idx]
        for aug in self.augmentations:
            x = aug(x)
        return x, self.y[idx]


def _default_base_url(archive: str) -> str:
    archive = archive.upper()
    if archive == "UCR":
        return "https://www.timeseriesclassification.com/Downloads"
    if archive == "UEA":
        return "https://www.timeseriesclassification.com/Downloads"
    raise ValueError(f"Unknown archive '{archive}'. Use 'UCR' or 'UEA'.")


def _headers_for_url(url: str) -> Dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
    }
    if "zenodo.org" in url:
        headers["Referer"] = "https://zenodo.org/"
        match = re.search(r"zenodo\\.org/records/(\\d+)", url)
        if match:
            headers["Referer"] = f"https://zenodo.org/records/{match.group(1)}"
    return headers


def _is_temporary_http_error(code: int) -> bool:
    return code in (429, 500, 502, 503, 504)


def _download_file(url: str, path: str) -> None:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            safe_url = _sanitize_url(url)
            req = Request(safe_url, headers=_headers_for_url(safe_url))
            with urlopen(req, timeout=60) as resp, open(path, "wb") as f:
                f.write(resp.read())
            return
        except HTTPError as exc:
            last_error = exc
            if _is_temporary_http_error(exc.code):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except URLError as exc:
            last_error = exc
            time.sleep(1.0 * (attempt + 1))
            continue
    if last_error is not None:
        raise last_error


def _build_file_url(files_base_url: str, filename: str) -> str:
    if "{file}" in files_base_url:
        return files_base_url.format(file=filename)
    return f"{files_base_url.rstrip('/')}/{filename}"


def _expand_zenodo_urls(url: str) -> List[str]:
    if "zenodo.org/records/" not in url:
        return [url]
    urls = [url]
    if "download=1" in url:
        urls.append(url.replace("download=1", "download=0"))
    else:
        sep = "&" if "?" in url else "?"
        urls.append(f"{url}{sep}download=1")
    # de-dup while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _normalize_files_base_urls(files_base_url: Optional[Union[str, List[str]]]) -> List[str]:
    if files_base_url is None:
        return []
    if isinstance(files_base_url, list):
        return [str(u) for u in files_base_url if str(u).strip()]
    return [str(files_base_url)]


def _sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    safe_path = quote(parts.path, safe="/%")
    safe_query = quote(parts.query, safe="=&%")
    return urlunsplit((parts.scheme, parts.netloc, safe_path, safe_query, parts.fragment))


def _extract_zenodo_record_id(urls: List[str]) -> Optional[str]:
    for url in urls:
        match = re.search(r"zenodo\\.org/(?:records|record)/(\\d+)", url)
        if match:
            return match.group(1)
    return None


def _default_zenodo_record_id(dataset_name: str, archive: str) -> Optional[str]:
    if archive.upper() == "UEA":
        per_dataset = {
            "Handwriting": "11206227",
        }
        return per_dataset.get(dataset_name)
    return None


def _download_from_zenodo_record(record_id: str, filename: str, path: str) -> None:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    req = Request(api_url, headers=_headers_for_url(api_url))
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    files = payload.get("files", [])
    for item in files:
        if item.get("key") == filename:
            links = item.get("links", {})
            download_url = links.get("download") or links.get("self")
            if download_url:
                _download_file(download_url, path)
                return
    raise ValueError(f"File '{filename}' not found in Zenodo record {record_id}.")


def _download_from_zenodo_zip_record(record_id: str, filename: str, path: str) -> None:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    req = Request(api_url, headers=_headers_for_url(api_url))
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    files = payload.get("files", [])
    for item in files:
        if item.get("key") == filename:
            links = item.get("links", {})
            download_url = links.get("download") or links.get("self")
            if download_url:
                _download_file(download_url, path)
                return
    raise ValueError(f"File '{filename}' not found in Zenodo record {record_id}.")


def _extract_dataset_from_zip(zip_path: str, data_dir: str, dataset_name: str) -> str:
    extract_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        dataset_prefix = f"{dataset_name}/"
        matched = [m for m in members if m.startswith(dataset_prefix) or m.endswith(f"/{dataset_name}_TRAIN.ts") or m.endswith(f"/{dataset_name}_TEST.ts")]
        if not matched:
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found inside {zip_path}")
        for member in matched:
            if member.endswith("/"):
                continue
            zf.extract(member, data_dir)
    return extract_dir


def _download_with_fallback(urls: List[str], path: str) -> None:
    last_error: Optional[Exception] = None
    for url in urls:
        try:
            _download_file(url, path)
            return
        except (HTTPError, URLError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error


def _download_dataset(
    data_dir: str,
    dataset_name: str,
    archive: str,
    base_url: Optional[str],
    files_base_url: Optional[Union[str, List[str]]],
    zenodo_record_id: Optional[str],
    allow_full_archive_download: bool,
) -> str:
    os.makedirs(data_dir, exist_ok=True)
    extract_dir = os.path.join(data_dir, dataset_name)
    train_path = os.path.join(extract_dir, f"{dataset_name}_TRAIN.ts")
    test_path = os.path.join(extract_dir, f"{dataset_name}_TEST.ts")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return extract_dir

    zenodo_error: Optional[Exception] = None
    effective_record_id = zenodo_record_id or _default_zenodo_record_id(dataset_name, archive)
    if effective_record_id:
        os.makedirs(extract_dir, exist_ok=True)
        try:
            _download_from_zenodo_record(effective_record_id, f"{dataset_name}_TRAIN.ts", train_path)
            _download_from_zenodo_record(effective_record_id, f"{dataset_name}_TEST.ts", test_path)
            return extract_dir
        except (HTTPError, URLError, ValueError) as exc:
            zenodo_error = exc
            if os.path.exists(train_path) and not os.path.getsize(train_path):
                os.remove(train_path)
            if os.path.exists(test_path) and not os.path.getsize(test_path):
                os.remove(test_path)

    files_base_urls = _normalize_files_base_urls(files_base_url)
    ts_error: Optional[Exception] = None
    if files_base_urls:
        os.makedirs(extract_dir, exist_ok=True)
        train_urls: List[str] = []
        test_urls: List[str] = []
        for base in files_base_urls:
            train_urls.extend(_expand_zenodo_urls(_build_file_url(base, f"{dataset_name}_TRAIN.ts")))
            test_urls.extend(_expand_zenodo_urls(_build_file_url(base, f"{dataset_name}_TEST.ts")))
        try:
            _download_with_fallback(train_urls, train_path)
            _download_with_fallback(test_urls, test_path)
            return extract_dir
        except (HTTPError, URLError, ValueError) as exc:
            ts_error = exc
            if os.path.exists(train_path) and not os.path.getsize(train_path):
                os.remove(train_path)
            if os.path.exists(test_path) and not os.path.getsize(test_path):
                os.remove(test_path)
        record_id = _extract_zenodo_record_id(files_base_urls)
        if record_id:
            try:
                _download_from_zenodo_record(record_id, f"{dataset_name}_TRAIN.ts", train_path)
                _download_from_zenodo_record(record_id, f"{dataset_name}_TEST.ts", test_path)
                return extract_dir
            except (HTTPError, URLError, ValueError) as exc:
                ts_error = exc

    url_base = base_url or _default_base_url(archive)
    zip_name = f"{dataset_name}.zip"
    url = f"{url_base}/{zip_name}"
    zip_path = os.path.join(data_dir, zip_name)
    if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
    zip_error: Optional[Exception] = None
    if not os.path.exists(zip_path):
        try:
            _download_file(url, zip_path)
        except (HTTPError, URLError, ValueError) as exc:
            zip_error = exc
    if zip_error is None and os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
        zip_error = ValueError("Downloaded file is not a zip.")

    if zip_error is not None and archive.upper() == "UEA" and allow_full_archive_download:
        full_archive_zip = os.path.join(data_dir, "TSML_MV_Archive_2018.zip")
        try:
            if not os.path.exists(full_archive_zip):
                _download_from_zenodo_zip_record("11206331", "TSML MV Archive 2018.zip", full_archive_zip)
            return _extract_dataset_from_zip(full_archive_zip, data_dir, dataset_name)
        except Exception as exc:
            zip_error = exc

    if zip_error is not None:
        message = ""
        if os.path.exists(zip_path):
            try:
                with open(zip_path, "rb") as f:
                    message = f.read(200).decode("utf-8", errors="ignore").strip()
            except Exception:
                message = ""
        hint = "Set 'files_base_url' to a direct .ts mirror or try downloading manually."
        zenodo_extra = f" Also failed via Zenodo API: {zenodo_error}." if zenodo_error is not None else ""
        ts_extra = f" Also failed to download .ts files: {ts_error}." if ts_error is not None else ""
        raise ValueError(
            f"Failed to download zip from {url}. Error: {zip_error}.{zenodo_extra}{ts_extra} "
            f"{'Message: ' + message if message else hint}"
        )
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
    return extract_dir


def _parse_ts_file(path: str) -> Tuple[List[np.ndarray], List[Any]]:
    data: List[np.ndarray] = []
    labels: List[Any] = []
    in_data = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if lower.startswith("@data"):
                in_data = True
                continue
            if not in_data or line.startswith("@"):
                continue

            parts = line.split(":")
            if len(parts) < 2:
                continue
            *series_parts, label = parts
            dims: List[np.ndarray] = []
            for series in series_parts:
                values = [float(v) if v != "?" else np.nan for v in series.split(",") if v != ""]
                dims.append(np.asarray(values, dtype=np.float32))
            sample = np.stack(dims, axis=0)
            data.append(sample)
            labels.append(label.strip().strip("'").strip('"'))
    return data, labels


def _pad_and_impute(series_list: List[np.ndarray], pad_value: float) -> np.ndarray:
    max_len = max(s.shape[1] for s in series_list)
    num_channels = series_list[0].shape[0]
    out = np.full((len(series_list), num_channels, max_len), pad_value, dtype=np.float32)
    for i, s in enumerate(series_list):
        out[i, :, : s.shape[1]] = s
    # Replace NaNs with per-channel mean
    for c in range(num_channels):
        channel = out[:, c, :]
        mask = np.isnan(channel)
        if mask.any():
            mean_val = np.nanmean(channel)
            channel[mask] = mean_val if not np.isnan(mean_val) else 0.0
            out[:, c, :] = channel
    return out


def _normalize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "per_series":
        def _norm(x: np.ndarray) -> np.ndarray:
            mean = x.mean(axis=2, keepdims=True)
            std = x.std(axis=2, keepdims=True) + 1e-6
            return (x - mean) / std

        return _norm(x_train), _norm(x_val), _norm(x_test)

    if mode == "global":
        mean = x_train.mean(axis=(0, 2), keepdims=True)
        std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
        return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std

    return x_train, x_val, x_test


def _semantic_ood_split(
    x: np.ndarray,
    y: np.ndarray,
    id_labels: List[Any],
    ood_labels: List[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    is_id = np.isin(y, id_labels)
    y_binary = np.where(is_id, 0, 1)
    return x, y_binary


def _balance_binary_split(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx_id = np.where(y == 0)[0]
    idx_ood = np.where(y == 1)[0]
    if len(idx_id) == 0 or len(idx_ood) == 0:
        return x, y
    n = min(len(idx_id), len(idx_ood))
    rng = np.random.default_rng(seed)
    sel_id = rng.choice(idx_id, size=n, replace=False)
    sel_ood = rng.choice(idx_ood, size=n, replace=False)
    sel = np.concatenate([sel_id, sel_ood])
    rng.shuffle(sel)
    return x[sel], y[sel]


def _split_train_val(x: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(x, y, test_size=val_ratio, random_state=seed, stratify=y)


@register("ucr")
@register("ucr_uea")
def load_ucr_uea(
    data_dir: str,
    dataset_name: str,
    archive: str = "UCR",
    base_url: Optional[str] = None,
    files_base_url: Optional[Union[str, List[str]]] = None,
    zenodo_record_id: Optional[str] = None,
    allow_full_archive_download: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
    normalize: str = "per_series",
    pad_value: float = 0.0,
    augmentations: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
) -> Dict[str, Any]:
    cfg = UCRUEAConfig(
        data_dir=data_dir,
        dataset_name=dataset_name,
        archive=archive,
        base_url=base_url,
        files_base_url=files_base_url,
        zenodo_record_id=zenodo_record_id,
        allow_full_archive_download=allow_full_archive_download,
        val_ratio=val_ratio,
        seed=seed,
        normalize=normalize,
        pad_value=pad_value,
        augmentations=augmentations,
    )
    root = _download_dataset(
        cfg.data_dir,
        cfg.dataset_name,
        cfg.archive,
        cfg.base_url,
        cfg.files_base_url,
        cfg.zenodo_record_id,
        cfg.allow_full_archive_download,
    )
    train_path = os.path.join(root, f"{cfg.dataset_name}_TRAIN.ts")
    test_path = os.path.join(root, f"{cfg.dataset_name}_TEST.ts")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find .ts files under {root}")

    train_series, train_labels = _parse_ts_file(train_path)
    test_series, test_labels = _parse_ts_file(test_path)
    x_train = _pad_and_impute(train_series, cfg.pad_value)
    x_test = _pad_and_impute(test_series, cfg.pad_value)
    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)

    def _label_sort_key(lbl: Any) -> tuple:
        try:
            return (0, float(lbl))
        except (ValueError, TypeError):
            return (1, str(lbl))

    unique_labels = sorted(set(y_train.tolist()) | set(y_test.tolist()), key=_label_sort_key)
    mid = max(1, len(unique_labels) // 2)
    id_labels = unique_labels[:mid]
    ood_labels = unique_labels[mid:]

    x_train_split, x_val_split, y_train_split, y_val_split = _split_train_val(
        x_train, y_train, cfg.val_ratio, cfg.seed
    )
    id_mask = np.isin(y_train_split, id_labels)
    x_train_id = x_train_split[id_mask]
    id_map = {label: idx for idx, label in enumerate(id_labels)}
    y_train_id = np.asarray([id_map[label] for label in y_train_split[id_mask]], dtype=np.int32)

    x_val_all, y_val_all = _semantic_ood_split(x_val_split, y_val_split, id_labels, ood_labels)
    x_test_all, y_test_all = _semantic_ood_split(x_test, y_test, id_labels, ood_labels)

    x_val_all, y_val_all = _balance_binary_split(x_val_all, y_val_all, cfg.seed + 1)
    x_test_all, y_test_all = _balance_binary_split(x_test_all, y_test_all, cfg.seed + 2)

    x_train_id, x_val_all, x_test_all = _normalize(x_train_id, x_val_all, x_test_all, cfg.normalize)

    return {
        "train": {"x": x_train_id, "y": y_train_id},
        "val": {"x": x_val_all, "y": y_val_all},
        "test": {"x": x_test_all, "y": y_test_all},
        "metadata": {
            "dataset_name": cfg.dataset_name,
            "archive": cfg.archive,
            "id_labels": id_labels,
            "ood_labels": ood_labels,
            "id_label_map": id_map,
            "data_dir": cfg.data_dir,
        },
    }
