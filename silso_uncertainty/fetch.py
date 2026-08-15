"""
Download and cache SILSO sunspot-number data files.

SILSO serves plain files over HTTPS with no versioning in the URL: the same URL
always returns the current release, which is revised as provisional values become
definitive. That means the URL is *stable* but the *content* is not. We therefore
record an SHA-256 and the download time alongside every cached file, so that any
result computed from a cache can be tied to a specific byte-level snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://www.sidc.be/SILSO/DATA"

#: Files this prototype knows how to fetch.
FILES = {
    "daily_total": "SN_d_tot_V2.0.csv",
    "daily_hemispheric": "SN_d_hem_V2.0.csv",
    "monthly_total": "SN_m_tot_V2.0.csv",
    "monthly_smoothed": "SN_ms_tot_V2.0.csv",
    "yearly_total": "SN_y_tot_V2.0.csv",
}


def default_cache_dir() -> Path:
    """Cache location, overridable with ``SILSO_CACHE``."""
    return Path(os.environ.get("SILSO_CACHE", Path.home() / ".silso_cache"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, cache_dir: Path | None = None, refresh: bool = False,
          timeout: float = 120.0) -> Path:
    """
    Return a local path to the SILSO file named by ``key``, downloading if needed.

    Parameters
    ----------
    key : str
        One of the keys of `FILES`.
    refresh : bool
        Re-download even if a cached copy exists.

    Notes
    -----
    The download is written to a temporary file and only moved into place on
    success, so an interrupted run cannot leave a truncated file in the cache
    that later looks valid. Earlier hand-rolled fetches of these URLs returned
    zero bytes because they did not follow redirects; ``urllib`` follows them.
    """
    if key not in FILES:
        raise KeyError(f"unknown SILSO file {key!r}; known keys: {sorted(FILES)}")

    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / FILES[key]
    meta_path = target.with_suffix(target.suffix + ".meta.json")

    if target.exists() and not refresh:
        return target

    url = f"{BASE_URL}/{FILES[key]}"
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as fh:
        if resp.status != 200:
            raise OSError(f"{url} returned HTTP {resp.status}")
        fh.write(resp.read())

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise OSError(f"{url} returned an empty body")

    tmp.replace(target)
    meta_path.write_text(json.dumps({
        "url": url,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
    }, indent=2))
    return target


def provenance(key: str, cache_dir: Path | None = None) -> dict:
    """Return the recorded provenance for a cached file, or ``{}`` if absent."""
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    meta_path = (cache_dir / FILES[key]).with_suffix(".csv.meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}
