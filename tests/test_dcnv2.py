import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import pytest

from ctr.dcnv2 import CrossLayer, OffsetCats, build_dcnv2


def _cross_numpy(x0, xl, W, b):
    return x0 * (xl @ W + b) + xl


def test_full_rank_cross_matches_numpy_reference():
    rng = np.random.default_rng(0)
    d, B = 8, 5
    x0, xl = rng.normal(size=(B, d)).astype("float32"), rng.normal(size=(B, d)).astype("float32")
    layer = CrossLayer()
    layer([x0, xl])
    W, b = rng.normal(size=(d, d)).astype("float32"), rng.normal(size=(d,)).astype("float32")
    layer.w.set_weights([W, b])
    np.testing.assert_allclose(layer([x0, xl]).numpy(), _cross_numpy(x0, xl, W, b), rtol=1e-5, atol=1e-5)


def test_low_rank_cross_matches_numpy_reference():
    rng = np.random.default_rng(1)
    d, r, B = 8, 3, 5
    x0, xl = rng.normal(size=(B, d)).astype("float32"), rng.normal(size=(B, d)).astype("float32")
    layer = CrossLayer(rank=r)
    layer([x0, xl])
    V, U = rng.normal(size=(d, r)).astype("float32"), rng.normal(size=(r, d)).astype("float32")
    b = rng.normal(size=(d,)).astype("float32")
    layer.v.set_weights([V])
    layer.u.set_weights([U, b])
    np.testing.assert_allclose(layer([x0, xl]).numpy(), _cross_numpy(x0, xl, V @ U, b), rtol=1e-5, atol=1e-5)


def test_zero_weights_make_cross_layer_identity_on_xl():
    d = 6
    x0 = np.random.default_rng(2).normal(size=(4, d)).astype("float32")
    xl = np.random.default_rng(3).normal(size=(4, d)).astype("float32")
    layer = CrossLayer()
    layer([x0, xl])
    layer.w.set_weights([np.zeros((d, d), "float32"), np.zeros(d, "float32")])
    np.testing.assert_allclose(layer([x0, xl]).numpy(), xl, atol=1e-7)


def test_cross_degree_is_polynomial_two_layers_give_cubic_terms():
    # With W=I, b=0:  x1 = x0*x0 + x0 ; x2 = x0*(x1) + x1 -> contains x0^3 (degree 3 crosses)
    d = 4
    x0 = np.array([[1.0, 2.0, 3.0, 4.0]], "float32")
    layers_ = [CrossLayer() for _ in range(2)]
    x = x0
    for l in layers_:
        l([x0, x])
        l.w.set_weights([np.eye(d, dtype="float32"), np.zeros(d, "float32")])
        x = l([x0, x]).numpy()
    expected = x0 * (x0 * x0 + x0) + (x0 * x0 + x0)
    np.testing.assert_allclose(x, expected, rtol=1e-6)


def test_offsets_separate_features_into_disjoint_id_ranges():
    out = OffsetCats(n_cat=3, num_buckets=100)(np.array([[0, 99, 5]], "int32")).numpy()
    np.testing.assert_array_equal(out, [[0, 199, 205]])


@pytest.mark.parametrize("structure", ["parallel", "stacked"])
@pytest.mark.parametrize("rank", [None, 4])
def test_model_outputs_valid_probabilities(structure, rank):
    m = build_dcnv2(num_buckets=50, emb_dim=4, n_cross=2, cross_rank=rank, mlp_units=(16, 8), structure=structure)
    rng = np.random.default_rng(4)
    p = m.predict({"dense": rng.normal(size=(32, 13)).astype("float32"),
                   "cats": rng.integers(0, 50, size=(32, 26)).astype("int32")}, verbose=0)
    assert p.shape == (32, 1) and np.all((p > 0) & (p < 1)) and np.all(np.isfinite(p))


def test_low_rank_has_fewer_parameters_than_full_rank():
    kw = dict(num_buckets=50, emb_dim=4, n_cross=3, mlp_units=(16,))
    assert build_dcnv2(cross_rank=4, **kw).count_params() < build_dcnv2(cross_rank=None, **kw).count_params()


def test_invalid_structure_rejected():
    with pytest.raises(ValueError):
        build_dcnv2(structure="bogus")


def test_model_save_and_reload_roundtrip(tmp_path):
    m = build_dcnv2(num_buckets=50, emb_dim=4, n_cross=2, cross_rank=3, mlp_units=(8,))
    x = {"dense": np.random.default_rng(5).normal(size=(6, 13)).astype("float32"),
         "cats": np.random.default_rng(6).integers(0, 50, size=(6, 26)).astype("int32")}
    before = m.predict(x, verbose=0)
    path = str(tmp_path / "m.keras")
    m.save(path)
    after = keras.saving.load_model(path).predict(x, verbose=0)
    np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
