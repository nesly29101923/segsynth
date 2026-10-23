from __future__ import annotations

import sqlite3
import zlib

import numpy as np
import pytest

from segsynth.core.generator import run_to_folder
from segsynth.core.imageio import read_image, read_mask
from segsynth.core.labeltool import DEFECT_ID, LabelToolWriter, decode_rle, encode_rle
from segsynth.core.project import GenerationSettings


def test_rle_layout_empty_mask():
    blob = encode_rle(np.zeros((640, 220), np.uint8))
    ints = np.frombuffer(blob[:-4], "<i4").tolist()
    assert ints == [220, 640, 1, 0, 0]
    assert int.from_bytes(blob[-4:], "little") == zlib.crc32(blob[:-4])


def test_rle_layout_runs():
    m = np.zeros((10, 8), np.int32)
    m[2, 1:4] = 2
    m[2, 6:8] = 2
    m[5, 0] = 2
    ints = np.frombuffer(encode_rle(m)[:-4], "<i4").tolist()
    assert ints == [8, 10, 2, 0, 2, 3, 2, 1, 3, 2, 2, 6, 2, 2, 5, 0, 1, 2]


@pytest.mark.parametrize("seed", range(5))
def test_rle_roundtrip(seed):
    rng = np.random.default_rng(seed)
    m = (rng.random((97, 61)) > 0.8).astype(np.int32) * DEFECT_ID
    assert np.array_equal(decode_rle(encode_rle(m)), m)


def test_decode_rejects_bad_crc():
    blob = bytearray(encode_rle(np.ones((4, 4), np.int32) * 2))
    blob[-1] ^= 0xFF
    with pytest.raises(ValueError):
        decode_rle(bytes(blob))


def test_writer_schema_and_rows(tmp_path):
    db_path = tmp_path / "out.sqlite"
    mask = np.zeros((30, 20), np.uint8)
    mask[5:9, 3:12] = 1
    with LabelToolWriter(db_path, "scratch", "#AABBCC") as w:
        w.add("Images/a.png", mask)
        w.add("Images/b.png", np.zeros_like(mask))
    c = sqlite3.connect(db_path)
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    assert len(tables) == 20 and {"Samples", "SemanticSegmentationAnnotations"} <= tables
    assert c.execute("select SettingValue from Settings where SettingName='Version'").fetchone() == ("14",)
    assert c.execute("select AnnotationLabelName, AnnotationLabelColorHex from AnnotationLabelDefinitions where AnnotationLabelId=2").fetchone() == ("scratch", "#aabbcc")
    assert c.execute("select count(*) from AnnotationMembershipDefinitions").fetchone() == (7,)
    rows = c.execute("""select s.SampleFilePath, s.SampleImageRepresentationFilePath, s.SampleDatatype,
                        a.SemanticSegmentationRunLengthCode, a.LinkAnnotationMembershipId
                        from Samples s join SemanticSegmentationAnnotations a on a.LinkSampleId = s.SampleId
                        order by s.SampleFilePath""").fetchall()
    assert [r[0] for r in rows] == ["Images\\a.png", "Images\\b.png"]
    assert rows[0][1] == rows[0][0] and rows[0][2] == "Image" and rows[0][4] == 2
    assert np.array_equal(decode_rle(rows[0][3]) > 0, mask > 0)
    for table in ("ClassificationAnnotations", "AnomalyDetectionAnnotations"):
        assert c.execute(f"select count(*) from {table}").fetchone() == (2,)
    ids = c.execute("select SampleId from Samples").fetchall()
    assert all(i[0] == i[0].upper() and len(i[0]) == 36 for i in ids)


def test_generator_writes_database(small_project, tmp_path):
    small_project.name = "Synthetic set"
    small_project.settings.save_labeltool_db = True
    small_project.settings.save_masks = True
    out = tmp_path / "out"
    manifest = run_to_folder(small_project, out)
    assert manifest["labeltool_db"] == "Synthetic set.sqlite"
    assert len(list((out / "Images").glob("*.png"))) == 5
    c = sqlite3.connect(out / "Synthetic set.sqlite")
    rows = c.execute("""select s.SampleFilePath, a.SemanticSegmentationRunLengthCode from Samples s
                        join SemanticSegmentationAnnotations a on a.LinkSampleId = s.SampleId""").fetchall()
    assert len(rows) == 5
    for path, blob in rows:
        rel = path.replace("\\", "/")
        img = read_image(out / rel)
        labels = decode_rle(blob)
        assert labels.shape == img.shape[:2]
        mask = read_mask(out / "masks" / (rel.split("/")[-1]))
        assert np.array_equal(labels > 0, mask > 0)
        assert labels.max() == DEFECT_ID


def test_default_keeps_previous_layout(small_project, tmp_path):
    run_to_folder(small_project, tmp_path / "o")
    assert (tmp_path / "o" / "images").is_dir()
    assert not list((tmp_path / "o").glob("*.sqlite"))


def test_settings_roundtrip():
    s = GenerationSettings(save_labeltool_db=True, labeltool_class_name="dent", labeltool_class_color="#123456")
    assert GenerationSettings.from_dict(s.to_dict()) == s
