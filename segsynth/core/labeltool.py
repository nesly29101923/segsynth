"""Label tool database (.sqlite) export.

Writes a fresh database in the label tool's schema (settings version 14) with one
sample per synthetic image and its semantic-segmentation annotation.

Mask blob layout (little-endian int32):
    width, height, n_labels, label_ids[n_labels], n_runs,
    (row, col_start, length, label) * n_runs, crc32(all preceding bytes)
"""
from __future__ import annotations

import sqlite3
import time
import uuid
import zlib
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "14"
BACKGROUND_ID = 0
IGNORE_ID = 1
DEFECT_ID = 2
ANOMALY_NORMAL_ID = -1
ANOMALY_ANOMALOUS_ID = -2

MEMBERSHIPS = {
    "NONE": 1, "TRAIN": 2, "VALIDATE": 3, "TEST": 4,
    "FORCE_TRAIN": 5, "FORCE_VALIDATE": 6, "FORCE_TEST": 7,
}

TABLES = [
    "CREATE TABLE 'AIModels' ( 'AIModelId' TEXT PRIMARY KEY NOT NULL , 'ParentAIModelId' TEXT , 'AIModelCreationDate' TEXT NOT NULL , 'AIModelName' TEXT NOT NULL , 'AIModelUserDescription' TEXT , 'AIModelModelTypeName' TEXT NOT NULL , 'AIModelDeploymentFrameworkTypeName' TEXT NOT NULL , 'AIModelDeploymentFrameworkVersion' TEXT NOT NULL , 'AIModelJobName' TEXT , 'AIModelBuildNumber' INTEGER , 'AIModelArtefactsRootPath' TEXT )",
    "CREATE TABLE 'AnnotationLabelDefinitions' ( 'AnnotationLabelId' INTEGER PRIMARY KEY , 'AnnotationLabelName' TEXT NOT NULL , 'AnnotationLabelSemantics' TEXT NOT NULL , 'AnnotationLabelColorHex' TEXT NOT NULL )",
    "CREATE TABLE 'AnnotationMembershipDefinitions' ( 'AnnotationMembershipId' INTEGER PRIMARY KEY , 'AnnotationMembershipName' TEXT UNIQUE NOT NULL )",
    "CREATE TABLE 'AnomalyDetectionAnnotations' ( 'AnnotationId' TEXT PRIMARY KEY NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'LinkAnnotationLabelId' INTEGER NOT NULL , 'LinkAnnotationMembershipId' INTEGER NOT NULL )",
    "CREATE TABLE 'AnomalyDetectionOutputs' ( 'AnomalyDetectionOutputId' TEXT PRIMARY KEY NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'LinkAIModelId' TEXT NOT NULL , 'LinkTrainRunAnnotationLabelId' INTEGER , 'LinkTrainRunAnnotationMembershipId' INTEGER )",
    "CREATE TABLE 'AnomalyDetectionOutputsValues' ( 'AnomalyDetectionOutputValueId' TEXT PRIMARY KEY NOT NULL , 'LinkAnomalyDetectionOutputId' TEXT NOT NULL , 'AnomalyValue' REAL NOT NULL , 'AnomalyOutputImageFilePath' TEXT )",
    "CREATE TABLE 'ClassificationAnnotations' ( 'AnnotationId' TEXT PRIMARY KEY NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'LinkAnnotationLabelId' INTEGER NOT NULL , 'LinkAnnotationMembershipId' INTEGER NOT NULL )",
    "CREATE TABLE 'ClassificationOutputs' ( 'ClassificationOutputId' TEXT PRIMARY KEY NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'LinkAIModelId' TEXT NOT NULL , 'LinkTrainRunAnnotationLabelId' INTEGER , 'LinkTrainRunAnnotationMembershipId' INTEGER )",
    "CREATE TABLE 'ClassificationOutputsValues' ( 'ClassificationOutputValueId' TEXT PRIMARY KEY NOT NULL , 'LinkClassificationOutputId' TEXT NOT NULL , 'LinkClassValueAnnotationLabelId' INTEGER NOT NULL , 'ClassValue' REAL NOT NULL , 'ClassUncertainty' REAL , 'ClassOutputImageFilePath' TEXT )",
    "CREATE TABLE 'Groups' ( 'GroupId' TEXT PRIMARY KEY NOT NULL , 'GroupName' TEXT NOT NULL , 'GroupDescription' TEXT )",
    "CREATE TABLE 'PropertyCategoryDefinitions' ( 'PropertyCategoryId' TEXT PRIMARY KEY NOT NULL , 'PropertyCategoryName' TEXT UNIQUE NOT NULL , 'PropertyCategorySemantics' TEXT , 'PropertyCategoryDescription' TEXT )",
    "CREATE TABLE 'PropertyDefinitions' ( 'PropertyId' TEXT PRIMARY KEY NOT NULL , 'LinkPropertyCategoryId' TEXT NOT NULL , 'PropertyName' TEXT NOT NULL , 'PropertyDescription' TEXT , 'PropertyValueType' TEXT NOT NULL )",
    "CREATE TABLE 'SampleProperties' ( 'SamplePropertyId' TEXT PRIMARY KEY NOT NULL , 'LinkPropertyId' TEXT NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'PropertyValue' TEXT NOT NULL , 'OptionalLinkAIModelId' TEXT )",
    "CREATE TABLE 'SampleToGroupConnections' ( 'ConnectionId' TEXT PRIMARY KEY NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'LinkGroupId' TEXT NOT NULL )",
    "CREATE TABLE 'Samples' ( 'SampleId' TEXT PRIMARY KEY NOT NULL , 'SampleFilePath' TEXT , 'SampleImageRepresentationFilePath' TEXT , 'SampleCreationDate' TEXT NOT NULL , 'SampleAddedToDatabaseDate' TEXT NOT NULL , 'SampleMD5CheckSum' TEXT , 'SampleDatatype' TEXT NOT NULL )",
    "CREATE TABLE 'SemanticSegmentationAnnotations' ( 'AnnotationId' TEXT PRIMARY KEY NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'SemanticSegmentationRunLengthCode' BLOB , 'LinkAnnotationMembershipId' INTEGER NOT NULL )",
    "CREATE TABLE 'SemanticSegmentationOutputs' ( 'SemanticSegmentationOutputId' TEXT PRIMARY KEY NOT NULL , 'LinkSampleId' TEXT NOT NULL , 'LinkAIModelId' TEXT NOT NULL , 'TrainRunSemanticSegmentationRunLengthCode' BLOB , 'LinkTrainRunAnnotationMembershipId' INTEGER )",
    "CREATE TABLE 'SemanticSegmentationOutputsValues' ( 'SemanticSegmentationOutputValueId' TEXT PRIMARY KEY NOT NULL , 'LinkSemanticSegmentationOutputId' TEXT NOT NULL , 'SemanticSegmentationRunLengthCode' BLOB , 'SegmentationOutputImageFilePath' TEXT )",
    "CREATE TABLE 'Settings' ( 'SettingId' INTEGER PRIMARY KEY NOT NULL , 'SettingName' TEXT UNIQUE NOT NULL , 'SettingValue' TEXT )",
    "CREATE TABLE 'VersionInfo' ( 'Version' INTEGER NOT NULL , 'AppliedOn' TEXT , 'Description' TEXT )",
]

TSNE_CATEGORY = ("7C1166C0-DB32-4FD1-B202-338DDE7A1C51", "TSNE", "", "")
TSNE_PROPERTIES = [
    ("4FD0992B-6801-474D-A48F-81AA8BB672D0", TSNE_CATEGORY[0], "XValue", "X value of a sample for the TSNE plot.", "REAL"),
    ("B6F3C5FA-71D7-490B-911F-EB788B001561", TSNE_CATEGORY[0], "YValue", "Y value of a sample for the TSNE plot.", "REAL"),
]


def encode_rle(labels: np.ndarray) -> bytes:
    """Encode an HxW integer label image (0 = background)."""
    labels = np.asarray(labels)
    h, w = labels.shape
    ids = np.unique(labels)
    if BACKGROUND_ID not in ids:
        ids = np.concatenate([[BACKGROUND_ID], ids])
    padded = np.zeros((h, w + 2), np.int64)
    padded[:, 1:-1] = labels
    change = np.diff(padded, axis=1) != 0
    runs = []
    for row in np.nonzero(change.any(axis=1))[0]:
        cols = np.nonzero(change[row])[0]
        line = labels[row]
        for start, end in zip(cols[:-1], cols[1:]):
            value = int(line[start])
            if value != BACKGROUND_ID:
                runs.append((int(row), int(start), int(end - start), value))
    body = np.array([w, h, len(ids), *ids.tolist(), len(runs)], dtype="<i4").tobytes()
    if runs:
        body += np.asarray(runs, dtype="<i4").tobytes()
    return body + np.uint32(zlib.crc32(body)).astype("<u4").tobytes()


def decode_rle(blob: bytes) -> np.ndarray:
    body, crc = blob[:-4], int.from_bytes(blob[-4:], "little")
    if zlib.crc32(body) != crc:
        raise ValueError("Mask checksum mismatch")
    a = np.frombuffer(body, "<i4")
    w, h, n = int(a[0]), int(a[1]), int(a[2])
    n_runs = int(a[3 + n])
    runs = a[4 + n:4 + n + 4 * n_runs].reshape(-1, 4)
    out = np.zeros((h, w), np.int32)
    for row, col, length, label in runs:
        out[row, col:col + length] = label
    return out


def _guid() -> str:
    return str(uuid.uuid4()).upper()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LabelToolWriter:
    """Creates a new label tool database and appends samples to it."""

    def __init__(
        self,
        path: str | Path,
        class_name: str = "defect",
        class_color: str = "#ff5c6c",
        membership: str = "TRAIN",
    ):
        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.membership = MEMBERSHIPS[membership]
        self.count = 0
        self.db = sqlite3.connect(str(self.path))
        cur = self.db.cursor()
        for sql in TABLES:
            cur.execute(sql)
        cur.executemany("INSERT INTO AnnotationLabelDefinitions VALUES (?,?,?,?)", [
            (-2, "Anomalous", "ANOMALY_DETECTION_ANOMALOUS", "#FF8000"),
            (-1, "Normal", "ANOMALY_DETECTION_NORMAL", "#008000"),
            (BACKGROUND_ID, "Background", "SEMANTIC_SEGMENTATION_GENERIC_CLASS", "#ff0000"),
            (IGNORE_ID, "None", "IGNORE_FOR_TRAINING", "#808080"),
            (DEFECT_ID, class_name or "defect", "SEMANTIC_SEGMENTATION_GENERIC_CLASS", class_color.lower()),
        ])
        cur.executemany("INSERT INTO AnnotationMembershipDefinitions VALUES (?,?)",
                        [(v, k) for k, v in MEMBERSHIPS.items()])
        cur.execute("INSERT INTO Settings VALUES (1, 'Version', ?)", (SCHEMA_VERSION,))
        cur.execute("INSERT INTO PropertyCategoryDefinitions VALUES (?,?,?,?)", TSNE_CATEGORY)
        cur.executemany("INSERT INTO PropertyDefinitions VALUES (?,?,?,?,?)", TSNE_PROPERTIES)
        self.db.commit()

    def add(self, rel_path: str, mask: np.ndarray) -> str:
        """Add one image (path relative to the database folder) with its defect mask (non-zero = defect).

        Images without defect pixels are labeled good (anomaly "Normal"), others bad ("Anomalous").
        """
        sample_id = _guid()
        path = rel_path.replace("/", "\\")
        stamp = _now()
        labels = np.where(np.asarray(mask) > 0, DEFECT_ID, BACKGROUND_ID)
        cur = self.db.cursor()
        cur.execute("INSERT INTO Samples VALUES (?,?,?,?,?,?,?)",
                    (sample_id, path, path, stamp, stamp, "", "Image"))
        cur.execute("INSERT INTO SemanticSegmentationAnnotations VALUES (?,?,?,?)",
                    (_guid(), sample_id, encode_rle(labels), self.membership))
        cur.execute("INSERT INTO ClassificationAnnotations VALUES (?,?,?,?)",
                    (_guid(), sample_id, IGNORE_ID, self.membership))
        cur.execute("INSERT INTO AnomalyDetectionAnnotations VALUES (?,?,?,?)",
                    (_guid(), sample_id, ANOMALY_ANOMALOUS_ID if labels.any() else ANOMALY_NORMAL_ID,
                     self.membership))
        self.count += 1
        if self.count % 200 == 0:
            self.db.commit()
        return sample_id

    def close(self) -> None:
        if self.db is not None:
            self.db.commit()
            self.db.close()
            self.db = None

    def __enter__(self) -> "LabelToolWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
