"""Fetch every dataset in PLAN.md §6.2. Resumable: re-run the same command after a failure.

    uv run --extra training python -m training.download [--only mswc,musan,...] [--langs cy,eu]

MSWC opus shards are streamed into packed stores instead of ~20M loose files:
    data/mswc/{lang}/{split}/{n}.bin            concatenated .opus files
    data/mswc/{lang}/{split}/{n}.parquet        word, clip, offset, length (written last = done)
    data/mswc/{lang}/splits/{train,dev,test}.csv
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import time
import traceback
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from training.paths import DATA, DOWNLOADS, LANGUAGES, LIBRISPEECH, MLS, MSWC, MUSAN, RIRS, SPLITS

MSWC_REPO = "MLCommons/ml_spoken_words"
MLS_REPO = "facebook/multilingual_librispeech"
MLS_LANGS = {"de": "german", "fr": "french", "es": "spanish", "it": "italian", "pl": "polish"}
# Mirrors in order of preference (the EU mirror measured ~2x faster than www from France).
OPENSLR_MIRRORS = ["https://openslr.elda.org", "https://www.openslr.org"]
OPENSLR = {
    "musan": (["17/musan.tar.gz"], MUSAN),
    "rirs": (["28/rirs_noises.zip"], RIRS),
    "librispeech": (["12/test-clean.tar.gz", "12/test-other.tar.gz"], LIBRISPEECH),
}
MIN_FREE_GB = 160


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- tasks


@dataclass(frozen=True)
class Shard:
    lang: str
    split: str
    n: int
    size: int
    sha256: str

    @property
    def repo_path(self) -> str:
        return f"data/opus/{self.lang}/{self.split}/audio/{self.n}.tar.gz"

    @property
    def index(self) -> Path:
        return MSWC / self.lang / self.split / f"{self.n}.parquet"


def fetch_shard(shard: Shard) -> str:
    """Download one tar.gz shard, verify it, pack its clips, delete the archive."""
    if shard.index.exists():
        return f"{shard.repo_path}: already done"
    started = time.time()
    archive = Path(hf_hub_download(MSWC_REPO, shard.repo_path, repo_type="dataset",
                                   local_dir=DOWNLOADS / "mswc"))
    if archive.stat().st_size != shard.size or sha256(archive) != shard.sha256:
        archive.unlink()
        raise RuntimeError(f"{shard.repo_path}: checksum mismatch, archive deleted")

    shard.index.parent.mkdir(parents=True, exist_ok=True)
    blob = shard.index.with_suffix(".bin")
    tmp_blob = blob.with_suffix(".bin.tmp")
    words, clips, offsets, lengths = [], [], [], []
    offset = 0
    with tarfile.open(archive, mode="r|gz") as tf, open(tmp_blob, "wb") as out:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".opus")):
                continue
            name = member.name.rsplit("/", 1)[-1][: -len(".opus")]
            word, sep, clip = name.partition("_common_voice_")
            if not sep:
                raise RuntimeError(f"{shard.repo_path}: unexpected member {member.name}")
            data = tf.extractfile(member).read()
            out.write(data)
            words.append(word)
            clips.append("common_voice_" + clip)
            offsets.append(offset)
            lengths.append(len(data))
            offset += len(data)
    tmp_blob.replace(blob)
    table = pa.table({
        "word": pa.array(words, pa.string()),
        "clip": pa.array(clips, pa.string()),
        "offset": pa.array(offsets, pa.int64()),
        "length": pa.array(lengths, pa.int32()),
    })
    tmp_index = shard.index.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_index)
    tmp_index.replace(shard.index)  # the done marker
    archive.unlink()
    mb = shard.size / 1e6
    return f"{shard.repo_path}: {len(words)} clips, {mb:.0f} MB in {time.time() - started:.0f}s"


def fetch_splits(lang: str) -> str:
    dest = MSWC / lang / "splits"
    if all((dest / f"{s}.csv").exists() for s in SPLITS):
        return f"{lang} splits: already done"
    archive = Path(hf_hub_download(MSWC_REPO, f"data/splits/{lang}/splits.tar.gz",
                                   repo_type="dataset", local_dir=DOWNLOADS / "mswc"))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            name = member.name.rsplit("/", 1)[-1]
            if member.isfile() and name in {f"{s}.csv" for s in SPLITS} | {"version.txt"}:
                (dest / name).write_bytes(tf.extractfile(member).read())
    archive.unlink()
    return f"{lang} splits: done"


def fetch_url(url: str, dest: Path, retries: int = 5) -> Path:
    """HTTP download with resume (Range) and retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as res:
                if have and res.status != 206:
                    have = 0  # server ignored the range: restart
                total = have + int(res.headers.get("Content-Length", 0))
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(res, f, 1 << 22)
            if total and part.stat().st_size != total:
                raise OSError(f"short download {part.stat().st_size} != {total}")
            part.replace(dest)
            return dest
        except OSError as err:
            log(f"{url}: attempt {attempt + 1} failed ({err}); retrying")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: giving up after {retries} attempts")


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")


def fetch_openslr(name: str) -> str:
    resources, target = OPENSLR[name]
    done = target / ".done"
    if done.exists():
        return f"{name}: already done"
    for resource in resources:
        dest = DOWNLOADS / "openslr" / resource.rsplit("/", 1)[-1]
        for mirror in OPENSLR_MIRRORS:  # a partial file resumes on the next mirror
            try:
                archive = fetch_url(f"{mirror}/resources/{resource}", dest, retries=3)
                break
            except RuntimeError as err:
                log(str(err))
        else:
            raise RuntimeError(f"{resource}: every mirror failed")
        extract(archive, target.parent)  # archives contain their own top-level folder
        archive.unlink()
    done.touch()
    return f"{name}: done"


def fetch_mls(lang: str) -> str:
    target = MLS / lang
    done = target / ".done"
    if done.exists():
        return f"mls {lang}: already done"
    folder = f"data/mls_{MLS_LANGS[lang]}"
    api = HfApi()
    for split in ("dev", "test"):
        files = [f.path for f in api.list_repo_tree(MLS_REPO, path_in_repo=f"{folder}/{split}",
                                                    repo_type="dataset", recursive=True)
                 if hasattr(f, "size")]
        for path in files:
            local = Path(hf_hub_download(MLS_REPO, path, repo_type="dataset",
                                         local_dir=DOWNLOADS / "mls"))
            if path.endswith(".tar.gz"):
                extract(local, target / split / "audio")
                local.unlink()
            else:
                (target / split).mkdir(parents=True, exist_ok=True)
                shutil.move(local, target / split / local.name)
    done.touch()
    return f"mls {lang}: done"


# --------------------------------------------------------------------------- main


def list_shards(langs: list[str]) -> list[Shard]:
    api = HfApi()
    shards = []
    for lang in langs:
        for split in SPLITS:
            for f in api.list_repo_tree(MSWC_REPO, path_in_repo=f"data/opus/{lang}/{split}/audio",
                                        repo_type="dataset"):
                n = int(f.path.rsplit("/", 1)[-1].split(".")[0])
                shards.append(Shard(lang, split, n, f.size, f.lfs.sha256))
    return shards


def run(task, *args) -> str:
    """Retry wrapper executed in the worker processes."""
    for attempt in range(3):
        try:
            return task(*args)
        except Exception:  # noqa: BLE001 - report and retry anything
            err = traceback.format_exc(limit=3)
            if attempt == 2:
                raise RuntimeError(err) from None
            time.sleep(15 * (attempt + 1))
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--only", default="mswc,musan,rirs,librispeech,mls",
                   help="comma list of: mswc, musan, rirs, librispeech, mls")
    p.add_argument("--langs", default=",".join(LANGUAGES))
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    only = set(args.only.split(","))
    langs = [lang for lang in LANGUAGES if lang in args.langs.split(",")]

    DATA.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(DATA).free / 1e9
    log(f"data root {DATA}, {free_gb:.0f} GB free")
    if free_gb < MIN_FREE_GB and "mswc" in only and len(langs) == len(LANGUAGES):
        log(f"need ~{MIN_FREE_GB} GB free for everything; aborting")
        return 1

    tasks: list[tuple] = []
    if "mswc" in only:
        tasks += [(fetch_splits, lang) for lang in langs]
        shards = list_shards(langs)
        todo = [s for s in shards if not s.index.exists()]
        log(f"mswc: {len(shards)} shards, {len(todo)} to fetch, "
            f"{sum(s.size for s in todo) / 1e9:.1f} GB")
        tasks += [(fetch_shard, s) for s in todo]  # LANGUAGES order = smallest first
    for name in ("musan", "rirs", "librispeech"):
        if name in only:
            tasks.append((fetch_openslr, name))
    if "mls" in only:
        tasks += [(fetch_mls, lang) for lang in langs if lang in MLS_LANGS]

    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, *task): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                log(f"[{i}/{len(tasks)}] {fut.result()}")
            except Exception as err:  # noqa: BLE001
                failures += 1
                log(f"[{i}/{len(tasks)}] FAILED {futures[fut][0].__name__}{futures[fut][1:]}: {err}")
    log(f"finished with {failures} failure(s)" + ("; re-run to resume" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
