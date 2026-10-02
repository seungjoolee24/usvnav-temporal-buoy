"""Build a small source-only ZIP for manual Colab upload, without Git or data.

Run from the workspace root: python -m training.package_colab
No file is uploaded. Regenerate after editing training code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "_source_manifest.json"


def source_files(root: Path) -> list[Path]:
    """Use an allowlist; no recursive scan of runs, user data, or credentials."""
    root = root.resolve()
    selected = [root / name for name in ("pyproject.toml", "VERSION.txt", ".gitignore")]
    selected.extend((root / "usvnav").glob("*.py"))
    selected.extend((root / "training").glob("*.py"))
    selected.extend((root / "training" / "courses").glob("*.json"))
    selected.extend(root / "training" / name for name in ("README.md", "cloud_training.md", ".gitignore"))
    selected.extend((root / "training" / "notebooks").glob("*.ipynb"))
    curriculum = root / "training" / "courses" / "temporal-buoy-v1"
    selected.extend(curriculum.glob("*.json"))
    selected.extend(curriculum / name for name in ("review.md", "model-architecture.png",
                    "environment-layouts.png", "topview-examples.png", "temporal-sequence.png",
                    "paired-layouts.png"))
    result = []
    for path in sorted(set(selected)):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Source must be a regular file inside workspace: {path.name}")
        result.append(path)
    return result


def build_bundle(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root / "training" / "cloud"):
        raise ValueError("Output ZIP must be inside training/cloud")
    if output.suffix.lower() != ".zip":
        raise ValueError("Output filename must end in .zip")
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Choose a new --out filename; existing bundles are preserved")
    files = source_files(root)
    relative_names = {path.relative_to(root).as_posix() for path in files}
    required = {"usvnav/__init__.py", "training/__init__.py",
                "training/train_rgb_navigation.py", "pyproject.toml", "VERSION.txt"}
    if missing := required - relative_names:
        raise FileNotFoundError(f"Training source is not ready: {sorted(missing)}")
    entries = {}
    contents = {}
    for path in files:
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        contents[name] = data
        entries[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = dict(format="usvnav-colab-source/1",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    files=entries,
                    excluded=".venv, .git, training/data, training/runs, credentials, caches")
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr(MANIFEST, manifest_bytes)
    manifest_path.write_bytes(manifest_bytes)
    # Read back every archived file. No environment, checkpoint, or image data is needed.
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(entries) | {MANIFEST}
        for name, item in entries.items():
            data = archive.read(name)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Bundle verification failed: {name}")
    return dict(zip=str(output), manifest=str(manifest_path), files=len(entries),
                bytes=output.stat().st_size, sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training" / "cloud" / "usvnav-colab-source.zip")
    args = parser.parse_args()
    print(json.dumps(build_bundle(ROOT, args.out), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
