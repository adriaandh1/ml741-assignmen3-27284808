"""Regenerate analysis twice, check tests and privacy, and archive verified evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

from ml741_assignment3.analysis import require
from ml741_assignment3.artifacts import file_sha256, write_json


def main():
    root = Path.cwd()
    logs = root / "tmp/final-verification"
    logs.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"],
        [sys.executable, "-m", "pip", "check"],
        [sys.executable, "scripts/run_final.py"],
    ]
    checks = []
    for i, command in enumerate(commands):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        (logs / f"check_{i}.txt").write_text(result.stdout + result.stderr)
        require(result.returncode == 0, f"check failed: {command}")
        checks.append({"command": command, "result": result.stdout.strip()})
    manifests = []
    for i in range(2):
        result = subprocess.run(
            [sys.executable, "scripts/analyse_final.py"],
            text=True,
            capture_output=True,
            check=True,
        )
        (logs / f"analysis_{i}.txt").write_text(result.stdout + result.stderr)
        manifests.append(
            json.loads((root / "results/final/analysis_manifest.json").read_text())
        )
    require(manifests[0] == manifests[1], "analysis regeneration not bitwise identical")
    # Tidy table integrity must be anchored in the verified collection.
    completion = json.loads((root / "results/final/completion_manifest.json").read_text())
    require(
        completion["verified_runs"] == 90 and not completion["failures"],
        "incomplete collection",
    )
    for relative, expected in completion["sha256"].items():
        require(file_sha256(root / relative) == expected, f"changed evidence: {relative}")
    for relative in (
        "data/raw/dry_bean.zip",
        "data/processed/dry_bean.npz",
        "results/final/runs/dry_bean/1001/all_class_baseline/predictions.csv",
    ):
        subprocess.run(
            [
                "git",
                "-C",
                str(root.parent),
                "check-ignore",
                "--quiet",
                f"assignment-three/{relative}",
            ],
            check=True,
        )
    # Contact sheets support visual review of every scientific figure, not just favourites.
    pictures = sorted((root / "results/figures/final").glob("*.png"))
    for start in range(0, len(pictures), 6):
        sheet = Image.new("RGB", (1500, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(pictures[start : start + 6]):
            picture = Image.open(path).convert("RGB")
            picture.thumbnail((735, 360))
            x, y = (i % 2) * 750, (i // 2) * 400
            sheet.paste(picture, (x + (750 - picture.width) // 2, y + 25))
            draw.text((x + 10, y + 5), path.stem, fill="black")
        sheet.save(logs / f"figure_contact_{start // 6 + 1}.png")
    write_json(
        root / "results/final/readiness_checks.json",
        {
            "checks": checks,
            "analysis_regeneration": "two bitwise-identical complete passes",
            "verified_runs": 90,
            "repeatability": json.loads(
                (root / "results/summaries/repeatability.json").read_text()
            ),
            "private_files_git_ignored": True,
            "figure_count": len(pictures),
            "report_source": "report/main.tex",
        },
    )
    # Same-filesystem recovery copy, not a substitute for off-device disaster recovery.
    backup = root / "tmp/final-evidence-v1.tar.gz"
    files = []
    for directory in ("src", "scripts", "tests", "configs", "docs", "data", "results"):
        files.extend(
            p
            for p in (root / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
    files += [root / "requirements.lock.txt", root / "pyproject.toml", root / "README.md"]
    # Re-running creates a new, explicitly numbered snapshot; never overwrite an archive.
    number = 1
    while backup.exists():
        number += 1
        backup = root / f"tmp/final-evidence-v1-{number}.tar.gz"
    with tarfile.open(backup, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    expected = {p.relative_to(root).as_posix(): file_sha256(p) for p in files}
    with tarfile.open(backup, "r:gz") as archive:
        for member in archive.getmembers():
            stream = archive.extractfile(member)
            require(stream is not None, "invalid archived member")
            require(
                hashlib.sha256(stream.read()).hexdigest() == expected[member.name],
                f"backup mismatch: {member.name}",
            )
    write_json(
        root / "results/final/backup_manifest.json",
        {
            "path": backup.relative_to(root).as_posix(),
            "sha256": file_sha256(backup),
            "files_verified": len(files),
            "bytes": backup.stat().st_size,
            "scope": "local recovery archive on same filesystem; not off-device backup",
        },
    )
    print(f"PASS: tests, privacy, 90 bundles, two identical analyses, {len(pictures)} figures")
    print(f"Verified recovery archive: {backup} ({len(files)} files)")


if __name__ == "__main__":
    main()
