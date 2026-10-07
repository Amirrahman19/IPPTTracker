"""Dataset registry helpers: where things live, what is on disk, and the size budget."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from .config import CONFIG_DIR, data_dir, load_yaml

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
HF_ROWS = "https://datasets-server.huggingface.co/rows"


def registry() -> dict:
    return load_yaml(CONFIG_DIR / "datasets.yaml")


def dataset_path(name: str) -> Path:
    return data_dir() / "datasets" / name


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def status() -> list[dict]:
    reg = registry()
    rows = []
    for name, d in reg["datasets"].items():
        p = dataset_path(name)
        rows.append({"name": name, "method": d["method"], "present": p.exists() and any(p.iterdir()),
                     "size_gb": round(dir_size_gb(p), 3), "path": str(p), "source": d["source"]})
    return rows


def budget_remaining_gb() -> float:
    return registry()["budget_gb"] - sum(r["size_gb"] for r in status())


def _use_system_certs() -> None:
    """Verify TLS against the OS trust store (the conda env's certifi bundle misses some roots here)."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file, so zipfile can pull single members out of a huge archive."""

    BLOCK = 4 << 20

    def __init__(self, url: str):
        import requests
        _use_system_certs()
        self._s = requests.Session()
        r = self._s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        r.raise_for_status()
        self.url = r.url                       # resolved CDN url, avoids a redirect per block
        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        r.close()
        self._pos = 0
        self._blk_start, self._blk = 0, b""

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, off: int, whence: int = io.SEEK_SET) -> int:
        self._pos = {io.SEEK_SET: off, io.SEEK_CUR: self._pos + off, io.SEEK_END: self.size + off}[whence]
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self._pos + n > self.size:
            n = self.size - self._pos
        out = bytearray()
        while n > 0:
            off = self._pos - self._blk_start
            if not (0 <= off < len(self._blk)):
                self._blk_start = self._pos
                end = min(self._pos + self.BLOCK, self.size) - 1
                r = self._s.get(self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=120)
                r.raise_for_status()
                self._blk, off = r.content, 0
            take = self._blk[off:off + n]
            out += take
            self._pos += len(take)
            n -= len(take)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def _hf_zip(d: dict, dest: Path) -> str:
    """Extract only the zip members under `prefix` from a zip hosted on the Hub."""
    f = HttpRangeFile(HF_RESOLVE.format(repo=d["hf_repo"], path=d["hf_file"]))
    prefix = d.get("prefix", "")
    with zipfile.ZipFile(f) as z:
        names = [i for i in z.infolist() if not i.is_dir() and i.filename.startswith(prefix)]
        need = sum(i.file_size for i in names) / 1e9
        if need > budget_remaining_gb():
            raise RuntimeError(f"{len(names)} files need {need:.2f} GB, over the remaining size budget")
        from tqdm import tqdm
        for info in tqdm(names, desc=d["hf_repo"], unit="file"):
            out = dest / Path(info.filename).name
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            with z.open(info) as src, open(out.with_suffix(".part"), "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            out.with_suffix(".part").replace(out)
    return f"extracted {len(names)} files ({need:.2f} GB) to {dest}"


def _hf_rows(d: dict, dest: Path) -> str:
    """Download selected classes from a Hub video dataset through the dataset-viewer API (no auth, no pyarrow)."""
    import requests
    from tqdm import tqdm
    _use_system_certs()
    wanted = {int(k): v for k, v in d["labels"].items()}       # class id -> short name
    found: list[tuple[str, int, int, str]] = []                # (split, row idx, label, url)
    for split in d["splits"]:
        off = 0
        while True:
            r = requests.get(HF_ROWS, params={"dataset": d["hf_repo"], "config": "default", "split": split,
                                              "offset": off, "length": 100}, timeout=120)
            r.raise_for_status()
            js = r.json()
            for row in js["rows"]:
                if row["row"]["label"] in wanted:
                    found.append((split, row["row_idx"], row["row"]["label"], row["row"]["video"]["src"]))
            off += 100
            if off >= js["num_rows_total"]:
                break
    for split, idx, label, url in tqdm(found, desc=d["hf_repo"], unit="clip"):
        out = dest / f"{wanted[label]}_{split}_{idx:05d}.avi"
        if not out.exists():
            out.write_bytes(requests.get(url, timeout=120).content)
    return f"downloaded {len(found)} clips to {dest}"


def fetch(name: str) -> str:
    """Download automatically when possible, otherwise return manual instructions."""
    d = registry()["datasets"][name]
    dest = dataset_path(name)
    dest.mkdir(parents=True, exist_ok=True)
    if d["method"] == "kaggle":
        if budget_remaining_gb() <= 0:
            raise RuntimeError("dataset size budget is used up; raise budget_gb in configs/datasets.yaml first")
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()  # needs ~/.kaggle/kaggle.json
        api.dataset_download_files(d["kaggle_slug"], path=str(dest), unzip=True)
        return f"downloaded {d['kaggle_slug']} to {dest}"
    if d["method"] == "hf_zip":
        return _hf_zip(d, dest)
    if d["method"] == "hf_rows":
        return _hf_rows(d, dest)
    return f"manual download: get it from {d['source']} and unpack into {dest}"
