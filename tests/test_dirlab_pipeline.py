from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.run_dirlab_pipeline import discover_case_archives


def _write_archive(path: Path, *, inspiratory: bool = True, expiratory: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        if inspiratory:
            archive.writestr("case/_iBHCT.img", b"insp")
        if expiratory:
            archive.writestr("case/_eBHCT.img", b"exp")


def test_discovers_archives_recursively_and_normalizes_case_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    first = data_dir / "raw_zips" / "COPD1.zip"
    second = data_dir / "downloads" / "copd2_pack.zip"
    _write_archive(first)
    _write_archive(second)

    found = discover_case_archives(data_dir)

    assert found == {"copd1": first, "copd2": second}


def test_rejects_duplicate_archives_for_same_case(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_archive(data_dir / "raw_zips" / "copd1.zip")
    _write_archive(data_dir / "backup" / "copd1_copy.zip")
    _write_archive(data_dir / "raw_zips" / "copd2.zip")

    with pytest.raises(ValueError, match="Multiple ZIPs match copd1"):
        discover_case_archives(data_dir)


def test_rejects_archive_without_required_expiratory_volume(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_archive(data_dir / "copd1.zip", expiratory=False)
    _write_archive(data_dir / "copd2.zip")

    with pytest.raises(ValueError, match="no DIR-Lab expiratory"):
        discover_case_archives(data_dir)


def test_requested_case_must_exist(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_archive(data_dir / "copd1.zip")
    _write_archive(data_dir / "copd2.zip")

    with pytest.raises(FileNotFoundError, match="copd3"):
        discover_case_archives(data_dir, ["copd1", "copd3"])
