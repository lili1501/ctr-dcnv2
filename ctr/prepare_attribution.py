"""Convert the Criteo Attribution Modeling dataset into the pipeline's raw format.

Source: https://huggingface.co/datasets/criteo/criteo-attribution-dataset  (criteo_attribution_dataset.tsv.gz,
16.5M ad impressions, tab-separated with a header, sorted by timestamp). License CC BY-NC-SA 4.0 (non-commercial,
attribution required). Cite: Diemert, Meynet, Lefortier, Galland, "Attribution Modeling Increases Efficiency of
Bidding in Display Advertising", AdKDD & TargetAd Workshop, KDD 2017 (arXiv:1707.06409).

Task: predict whether an impression is clicked (`click`) from what is known when the impression is served.

  label  <- click
  I1     <- time_since_last_click   (seconds; the dataset uses -1 when there is none; kept as is)
  C1     <- campaign
  C2     <- uid                      (a user id: user-level memorisation is possible; the split is chronological)
  C3..C11 <- cat1..cat9              (undisclosed contextual features)

Columns deliberately NOT used, because they are only known after the impression or are the outcome itself:
  timestamp (used only to check the order), conversion, conversion_timestamp, conversion_id, attribution,
  click_pos, click_nb (defined only for clicked impressions), cost, cpo (Criteo's transformed prices/outcomes).
`audit` below measures how strongly each excluded column is tied to the label, so the exclusion is evidence-based.

    python -m ctr.prepare_attribution --input data/raw/criteo_attribution_dataset.tsv.gz --output data/raw/criteo_attribution.txt
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os

import pandas as pd

# source column -> output position (output order is label, I1, C1, C2, C3..C11)
FEATURES = ["time_since_last_click", "campaign", "uid"] + [f"cat{i}" for i in range(1, 10)]
LABEL_SRC = "click"
LEAKY = ["conversion", "conversion_id", "attribution", "click_pos", "click_nb", "cost", "cpo"]
REQUIRED = ["timestamp", LABEL_SRC] + FEATURES


class _TolerantGzip(io.RawIOBase):
    """Reads a .gz and treats a cut-off stream (a partial download) as a normal end of file, recording that it happened."""

    def __init__(self, path):
        self.f = gzip.open(path, "rb")
        self.truncated = False

    def readable(self):
        return True

    def readinto(self, b):
        try:
            data = self.f.read1(len(b))
        except EOFError:
            self.truncated = True
            return 0
        b[:len(data)] = data
        return len(data)

    def close(self):
        self.f.close()
        super().close()


def _open(src):
    if src.endswith(".gz"):
        raw = _TolerantGzip(src)
        return io.BufferedReader(raw), raw
    return src, None


def _with_last_flag(reader):
    """Yield (chunk, is_last) so the final chunk can be treated specially (a cut-off download ends in a partial row)."""
    prev = None
    for c in reader:
        if prev is not None:
            yield prev, False
        prev = c
    if prev is not None:
        yield prev, True


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def convert(src: str, dst: str, chunksize: int = 1_000_000, max_rows: int | None = None) -> dict:
    """Stream src -> dst (tab-separated, no header, empty = missing). Returns the audit dict."""
    h, hraw = _open(src)
    head = pd.read_csv(h, sep="\t", nrows=0)
    if hraw is not None:
        h.close()
    missing = [c for c in REQUIRED if c not in head.columns]
    if missing:
        raise ValueError(f"{src}: missing expected columns {missing}; found {list(head.columns)}")
    leaky_present = [c for c in LEAKY if c in head.columns]
    usecols = REQUIRED + leaky_present

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".partial"
    rows = clicks = unsorted_steps = 0
    last_ts = None
    t_min = t_max = None
    # for each leaky column: click counts where the column is "present" (>= 1 for 0/1 flags, >= 0 for -1 sentinels) vs not
    stat = {c: {"n_present": 0, "clicks_present": 0, "n_absent": 0, "clicks_absent": 0} for c in leaky_present}
    first = True
    handle, traw = _open(src)
    with open(tmp, "w") as out:
        reader = pd.read_csv(handle, sep="\t", usecols=usecols, dtype=str, keep_default_na=False, chunksize=chunksize)
        for chunk, is_last in _with_last_flag(reader):
            if max_rows is not None and rows >= max_rows:
                break
            if is_last and traw is not None and traw.truncated:
                chunk = chunk.iloc[:-1]                  # the last line of a cut-off download may be incomplete
                if len(chunk) == 0:
                    break
            if max_rows is not None:
                chunk = chunk.iloc[: max_rows - rows]
            y = _num(chunk[LABEL_SRC])
            if y.isna().any() or not y.isin([0, 1]).all():
                raise ValueError("click column contains values other than 0/1")
            ts = _num(chunk["timestamp"])
            if ts.isna().any():
                raise ValueError("timestamp column contains non-numeric values")
            d = ts.diff().dropna()
            unsorted_steps += int((d < 0).sum())
            if last_ts is not None and ts.iloc[0] < last_ts:
                unsorted_steps += 1
            last_ts = ts.iloc[-1]
            t_min = ts.min() if t_min is None else min(t_min, ts.min())
            t_max = ts.max() if t_max is None else max(t_max, ts.max())

            for c in leaky_present:
                v = _num(chunk[c])
                present = (v >= 1) if c in ("conversion", "attribution") else (v >= 0)
                s = stat[c]
                s["n_present"] += int(present.sum()); s["clicks_present"] += int(y[present].sum())
                s["n_absent"] += int((~present).sum()); s["clicks_absent"] += int(y[~present].sum())

            o = pd.DataFrame({"label": y.astype(int)})
            for c in FEATURES:
                o[c] = chunk[c].to_numpy()
            o.to_csv(out, sep="\t", header=False, index=False)
            rows += len(chunk); clicks += int(y.sum())
            first = False
    truncated = bool(traw is not None and traw.truncated)
    if traw is not None:
        handle.close()
    if first or rows == 0:
        raise ValueError(f"{src}: no rows read")
    os.replace(tmp, dst)

    def rate(c, n):
        return None if n == 0 else c / n
    leakage = {c: {"rows_present": s["n_present"],
                   "click_rate_when_present": rate(s["clicks_present"], s["n_present"]),
                   "click_rate_when_absent": rate(s["clicks_absent"], s["n_absent"])} for c, s in stat.items()}
    return {"source": os.path.basename(src), "rows": rows, "click_rate": clicks / rows,
            "timestamp_min": float(t_min), "timestamp_max": float(t_max),
            "truncated_input": truncated, "max_rows": max_rows, "sorted_by_timestamp": unsorted_steps == 0, "unsorted_steps": unsorted_steps,
            "features_used": {"label": LABEL_SRC, "I1": FEATURES[0], "C1": FEATURES[1], "C2": FEATURES[2],
                              **{f"C{i + 3}": f"cat{i + 1}" for i in range(9)}},
            "excluded_columns": ["timestamp", *leaky_present], "leakage_check": leakage,
            "license": "CC BY-NC-SA 4.0 (non-commercial); cite Diemert et al., AdKDD 2017, arXiv:1707.06409"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/raw/criteo_attribution.txt")
    ap.add_argument("--audit", default=None, help="where to write the audit json (default: <output>.audit.json)")
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args(argv)
    audit = convert(a.input, a.output, max_rows=a.max_rows)
    path = a.audit or a.output + ".audit.json"
    with open(path, "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps({k: audit[k] for k in ("rows", "click_rate", "sorted_by_timestamp")}), "->", a.output)
    if not audit["sorted_by_timestamp"]:
        print("WARNING: input is not sorted by timestamp, so a positional split is NOT a time split. "
              "Sort it first (the dataset card says it is sorted).")
    return audit


if __name__ == "__main__":
    main()
