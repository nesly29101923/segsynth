from __future__ import annotations

import sqlite3

from segsynth.core.export import assign_splits, db_file_name, export_dataset
from segsynth.core.generator import run_to_folder


def _files(n_good, n_bad, kind="synthetic"):
    return ([{"file": f"{kind}/g{i}.png", "kind": kind, "label": "good"} for i in range(n_good)]
            + [{"file": f"{kind}/b{i}.png", "kind": kind, "label": "bad"} for i in range(n_bad)])


def test_split_is_stratified_and_reproducible():
    files = _files(40, 60)
    a = assign_splits(files, 15, seed=1)
    assert a == assign_splits(files, 15, seed=1)
    val = [f for f in files if a[f["file"]] == "VALIDATE"]
    assert sum(f["label"] == "good" for f in val) == 6
    assert sum(f["label"] == "bad" for f in val) == 9


def test_split_keeps_training_data():
    assert set(assign_splits(_files(1, 1), 50, 0).values()) == {"TRAIN"}
    assert list(assign_splits(_files(2, 0), 50, 0).values()).count("TRAIN") == 1
    assert set(assign_splits(_files(5, 5), 0, 0).values()) == {"TRAIN"}


def test_db_file_name_has_no_spaces():
    assert db_file_name("My dataset v2") == "My_dataset_v2.sqlite"
    assert db_file_name("  ") == "dataset.sqlite"


def test_workspace_is_readable_like_the_ai_tool(small_project, tmp_path):
    small_project.defects[0].count = 20
    staging = tmp_path / "staging"
    manifest = run_to_folder(small_project, staging, labeltool=False)
    dest = tmp_path / "ws"
    out = export_dataset(staging, dest, manifest, "Line 3 scratches", True, validate_percent=20, seed=3)
    assert sorted(p.name for p in dest.iterdir()) == ["DATA", "DATA_THUMB", "META", "manifest.json"]
    assert out["labeltool_db"] == "META/Line_3_scratches.sqlite"
    db = dest / "META" / "Line_3_scratches.sqlite"
    c = sqlite3.connect(db)
    rows = c.execute("""select s.SampleFilePath, a.LinkAnnotationMembershipId, cl.LinkAnnotationMembershipId,
                               an.LinkAnnotationMembershipId, an.LinkAnnotationLabelId
                        from Samples s
                        join SemanticSegmentationAnnotations a on a.LinkSampleId = s.SampleId
                        join ClassificationAnnotations cl on cl.LinkSampleId = s.SampleId
                        join AnomalyDetectionAnnotations an on an.LinkSampleId = s.SampleId""").fetchall()
    assert len(rows) == 22
    data_root = db.parent.parent / "DATA"
    for path, m1, m2, m3, _ in rows:
        assert "/" not in path and not path.startswith("\\")
        assert (data_root / path.replace("\\", "/")).is_file()
        assert m1 == m2 == m3 and m1 in (2, 3)
    splits = {(m1, lab) for _, m1, _, _, lab in rows}
    assert {(2, -1), (2, -2), (3, -2)} <= splits
    assert out["splits"]["VALIDATE"] >= 4 and out["splits"]["TRAIN"] + out["splits"]["VALIDATE"] == 22
    assert not (dest / "DATA" / ".labels").exists()


def test_plain_export_has_no_workspace(small_project, tmp_path):
    staging = tmp_path / "staging"
    manifest = run_to_folder(small_project, staging, labeltool=False)
    out = export_dataset(staging, tmp_path / "plain", manifest, "x", False)
    assert sorted(p.name for p in (tmp_path / "plain").iterdir()) == ["background", "manifest.json", "synthetic"]
    assert out["labeltool_db"] is None
