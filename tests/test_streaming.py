"""stream_split must read exactly the same rows as load_split, in the same order, without loading everything."""
import numpy as np
import pandas as pd
import pytest

from ctr.config import CAT_COLS, INT_COLS, LABEL
from ctr.data import count_rows, load_labels, load_split, stream_split

N = 5003  # deliberately not a multiple of the batch size


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    d = tmp_path_factory.mktemp("proc") / "train"
    d.mkdir()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({LABEL: rng.integers(0, 2, N).astype("int32")})
    for c in INT_COLS:
        df[c] = rng.normal(size=N).astype("float32")
    for c in CAT_COLS:
        df[c] = rng.integers(0, 1000, N).astype("int32")
    for i, lo in enumerate(range(0, N, 1700)):          # several part files, like Spark writes
        df.iloc[lo:lo + 1700].to_parquet(d / f"part-{i:05d}.parquet")
    return str(d.parent)


def _collect(gen):
    parts = list(gen)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def test_stream_matches_load_split_exactly_and_in_order(processed):
    ref = load_split(processed, "train")
    dense, cats, y = _collect(stream_split(processed, "train", batch_size=256))
    np.testing.assert_array_equal(y, ref["y"])
    np.testing.assert_array_equal(cats, ref["cats"])
    np.testing.assert_array_equal(dense, ref["dense"])


def test_batches_have_the_requested_size_except_possibly_the_last(processed):
    sizes = [len(b[2]) for b in stream_split(processed, "train", batch_size=256, shuffle_rows=2048)]
    assert sum(sizes) == N and max(sizes) <= 256 and sizes[0] == 256


def test_shuffle_keeps_the_same_rows_but_changes_order_and_is_seeded(processed):
    ref = load_split(processed, "train")
    _, c1, y1 = _collect(stream_split(processed, "train", 256, shuffle=True, shuffle_rows=1500, seed=1))
    _, c2, _ = _collect(stream_split(processed, "train", 256, shuffle=True, shuffle_rows=1500, seed=1))
    _, c3, _ = _collect(stream_split(processed, "train", 256, shuffle=True, shuffle_rows=1500, seed=2))
    assert not np.array_equal(c1, ref["cats"])                       # order changed
    key = lambda a: np.lexsort(a.T[::-1])
    np.testing.assert_array_equal(c1[key(c1)], ref["cats"][key(ref["cats"])])  # same multiset of rows
    np.testing.assert_array_equal(c1, c2)                            # same seed -> same order
    assert not np.array_equal(c1, c3)                                # different seed -> different order
    assert y1.sum() == ref["y"].sum()


def test_shuffle_is_local_to_a_chunk(processed):
    # rows never move farther than one chunk, so early rows stay near the start (chronology is kept coarsely)
    ref = load_split(processed, "train")
    _, c, _ = _collect(stream_split(processed, "train", 256, shuffle=True, shuffle_rows=1500, seed=0))
    first_chunk_ref = {tuple(r) for r in ref["cats"][:1500]}
    assert {tuple(r) for r in c[:1500]} == first_chunk_ref


def test_max_rows_and_count_and_label_reader(processed):
    _, _, y = _collect(stream_split(processed, "train", 256, max_rows=1234))
    assert len(y) == 1234
    assert count_rows(processed, "train") == N
    np.testing.assert_array_equal(load_labels(processed, "train", 777), load_split(processed, "train")["y"][:777])
