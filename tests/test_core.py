from __future__ import annotations

import numpy as np
import pytest

from segsynth.core import transform as tf
from segsynth.core.background import estimate_background
from segsynth.core.blend import BlendMode, blend
from segsynth.core.imageio import read_image, read_mask, write_image
from segsynth.core.patch import extract_patch, mask_bbox, soft_alpha
from segsynth.core.placement import PlacementMode, find_position
from segsynth.core.refine import deviation_map, refine_mask

from .conftest import add_scratch, striped_surface


class TestBackground:
    def test_vertical_estimate_recovers_stripes_under_defect(self, scratch_image):
        img, mask = scratch_image
        clean = striped_surface()
        bg = estimate_background(img, "vertical", 61, exclude_mask=mask)
        err = np.abs(bg[mask > 0] - clean[mask > 0].astype(np.float32)).mean()
        assert err < 8

    def test_grayscale_shape(self):
        img = striped_surface(channels=1)
        assert estimate_background(img, "isotropic", 31).shape == img.shape

    def test_none_mode_is_zero(self, scratch_image):
        assert not estimate_background(scratch_image[0], "none").any()


class TestPatch:
    def test_extract_shapes_and_ranges(self, scratch_image):
        img, mask = scratch_image
        p = extract_patch(img, mask, "vertical", 61, feather=3)
        h, w = p.mask.shape
        assert p.residual.shape == (h, w, 3) and p.pixels.shape == (h, w, 3)
        assert p.alpha.min() >= 0 and p.alpha.max() <= 1
        assert p.mask.sum() == mask.sum()
        assert p.residual[p.mask > 0].mean() > 60

    def test_empty_mask_raises(self, scratch_image):
        with pytest.raises(ValueError):
            extract_patch(scratch_image[0], np.zeros(scratch_image[1].shape, np.uint8))

    def test_soft_alpha_contains_mask(self):
        m = np.zeros((20, 20), np.uint8)
        m[8:12, 8:12] = 1
        a = soft_alpha(m, 3)
        assert (a[m > 0] == 1).all() and a[0, 0] < 0.05

    def test_bbox(self):
        m = np.zeros((10, 10), np.uint8)
        m[2:5, 3:7] = 1
        assert mask_bbox(m) == (3, 2, 7, 5)
        assert mask_bbox(np.zeros((3, 3))) is None


class TestTransform:
    def _patch(self, scratch_image):
        img, mask = scratch_image
        return extract_patch(img, mask, "vertical", 61)

    def test_identity_keeps_area(self, scratch_image):
        p = self._patch(scratch_image)
        t = tf.AppliedTransform(1, 1, 0, 0, 1, False, False)
        out = tf.apply(p, t, np.random.default_rng(0))
        assert abs(int(out.mask.sum()) - int(p.mask.sum())) / p.mask.sum() < 0.08

    def test_scale_grows_area(self, scratch_image):
        p = self._patch(scratch_image)
        t = tf.AppliedTransform(2, 1, 0, 0, 1, False, False)
        out = tf.apply(p, t, np.random.default_rng(0))
        assert 3.0 < out.mask.sum() / p.mask.sum() < 5.0

    def test_elongation_along_long_axis(self, scratch_image):
        p = self._patch(scratch_image)
        t = tf.AppliedTransform(1, 1.6, 0, 0, 1, False, False)
        out = tf.apply(p, t, np.random.default_rng(0))
        assert out.mask.shape[1] > p.mask.shape[1] * 1.35

    def test_all_layers_aligned(self, scratch_image):
        p = self._patch(scratch_image)
        rng = np.random.default_rng(3)
        out = tf.apply(p, tf.sample(tf.TransformParams(bend=(0.3, 0.3)), rng), rng, 4.0, 10.0)
        h, w = out.mask.shape
        assert out.residual.shape[:2] == (h, w) == out.alpha.shape == out.pixels.shape[:2]
        assert (out.alpha[out.mask > 0] > 0.3).mean() > 0.95

    def test_sampling_deterministic(self):
        params = tf.TransformParams()
        a = tf.sample(params, np.random.default_rng(11))
        b = tf.sample(params, np.random.default_rng(11))
        assert a == b

    def test_params_roundtrip(self):
        p = tf.TransformParams(scale=(0.5, 2.0), flip_h=0.5)
        assert tf.TransformParams.from_dict(p.to_dict()) == p


class TestPlacement:
    def test_inside_bounds(self):
        rng = np.random.default_rng(0)
        m = np.ones((20, 30), np.uint8)
        for _ in range(50):
            x, y = find_position(m, (100, 120), rng)
            assert 0 <= x <= 120 - 30 and 0 <= y <= 100 - 20

    def test_respects_occupied(self):
        rng = np.random.default_rng(0)
        m = np.ones((10, 10), np.uint8)
        occ = np.zeros((60, 60), np.uint8)
        occ[:, :40] = 1
        for _ in range(20):
            x, _ = find_position(m, (60, 60), rng, occupied=occ, min_gap=2)
            assert x >= 42

    def test_impossible_returns_none(self):
        m = np.ones((10, 10), np.uint8)
        assert find_position(m, (60, 60), np.random.default_rng(0), occupied=np.ones((60, 60), np.uint8), attempts=20) is None
        assert find_position(np.ones((80, 80), np.uint8), (60, 60), np.random.default_rng(0)) is None

    def test_keep_column(self):
        rng = np.random.default_rng(0)
        m = np.ones((10, 10), np.uint8)
        for _ in range(30):
            x, _ = find_position(m, (300, 300), rng, PlacementMode.KEEP_COLUMN, (150, 150), jitter=5)
            assert abs((x + 5) - 150) <= 6


class TestBlend:
    @pytest.mark.parametrize("mode", list(BlendMode))
    def test_modes_run_and_keep_dtype(self, scratch_image, mode):
        img, mask = scratch_image
        p = extract_patch(img, mask, "vertical", 61)
        target = striped_surface(seed=5)
        out = blend(target, p, (100, 60), mode)
        assert out.shape == target.shape and out.dtype == target.dtype
        assert (out != target).any()
        assert (out[:40] == target[:40]).all()

    def test_residual_preserves_target_texture(self, scratch_image):
        img, mask = scratch_image
        p = extract_patch(img, mask, "vertical", 61)
        target = striped_surface(seed=9)
        out = blend(target, p, (100, 60), BlendMode.RESIDUAL)
        ys, xs = np.nonzero(p.alpha < 0.01)
        assert np.abs(out[ys + 60, xs + 100].astype(int) - target[ys + 60, xs + 100].astype(int)).max() <= 1

    def test_uint16_and_gray(self):
        img, mask = add_scratch(striped_surface(channels=1, dtype=np.uint16))
        p = extract_patch(img, mask, "vertical", 61)
        target = striped_surface(seed=4, channels=1, dtype=np.uint16)
        for mode in BlendMode:
            out = blend(target, p, (90, 50), mode, 65535.0)
            assert out.dtype == np.uint16 and out.shape == target.shape


class TestRefine:
    def test_snaps_to_scratch(self, scratch_image):
        img, mask = scratch_image
        rough = np.zeros_like(mask)
        rough[100:170, 40:190] = 1
        refined = refine_mask(rough, deviation_map(img, "vertical", 61), 0.5)
        inter = (refined & mask).sum()
        union = (refined | mask).sum()
        assert inter / union > 0.6


class TestImageIO:
    def test_roundtrip_unicode_path(self, tmp_path):
        img = striped_surface(dtype=np.uint16)
        path = tmp_path / "görüntü_ş.png"
        write_image(path, img)
        back = read_image(path)
        assert back.dtype == np.uint16 and np.array_equal(back, img)

    def test_mask(self, tmp_path):
        m = np.zeros((5, 5), np.uint8)
        m[1:3, 1:3] = 255
        write_image(tmp_path / "m.png", m)
        assert read_mask(tmp_path / "m.png").sum() == 4

    def test_unreadable(self, tmp_path):
        (tmp_path / "bad.png").write_bytes(b"nope")
        with pytest.raises(IOError):
            read_image(tmp_path / "bad.png")
