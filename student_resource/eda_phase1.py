"""
Phase 1 EDA — Amazon ML Challenge 2026 (Business Entity Resolution)

Reads every train/test TSV once (tab-separated), reuses the frames, and writes
a full report covering:

  1. .shape of all source + ground-truth files
  2. Sample rows (head)
  3. Missing / empty values per column
  4. Country distribution (train vs test — France is test-only)
  5. Ground-truth match-count stats (singletons, histogram, outliers)
  6. Basic text stats (name/address length, empty address rates)
  7. Side-by-side ground-truth matched pairs (noise-pattern inspection)

Run from this directory:

    python eda_phase1.py

Output: eda_output.txt (same folder). Large files — expect a few minutes.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — resolved relative to this script so it works from any cwd
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(HERE, "dataset", "train")
TEST_DIR = os.path.join(HERE, "dataset", "test")
OUTPUT_PATH = os.path.join(HERE, "eda_output.txt")

TRAIN_FILES = {
    "train_source1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "train_source2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "train_source3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
    "train_ground_truth": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
}
TEST_FILES = {
    "test_source1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "test_source2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "test_source3": os.path.join(TEST_DIR, "test_source3.tsv"),
}

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
HEAD_N = 8
PAIR_EXAMPLES = 18
RNG_SEED = 42


class Tee:
    """Write to stdout and a UTF-8 file. Windows consoles are often cp1252."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                enc = getattr(s, "encoding", None) or "ascii"
                s.write(data.encode(enc, errors="replace").decode(enc, errors="replace"))
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def subbanner(title: str) -> None:
    print("\n" + "-" * 88)
    print(title)
    print("-" * 88)


def load_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Loading {os.path.basename(path):30s}  ({size_mb:7.1f} MB) ...", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True)
    print(f"    -> {df.shape[0]:,} rows × {df.shape[1]} cols", flush=True)
    return df


def is_blank(series: pd.Series) -> pd.Series:
    """True for NaN, None, or whitespace-only strings."""
    s = series.fillna("")
    return s.astype(str).str.strip().eq("")


def ascii_histogram(counts: pd.Series, max_bar: int = 50) -> None:
    """Print a simple text histogram. `counts` is value -> frequency."""
    if counts.empty:
        print("  (empty)")
        return
    peak = int(counts.max())
    peak = max(peak, 1)
    for key, freq in counts.items():
        bar_len = int(round(freq / peak * max_bar))
        print(f"  {str(key):>6} | {'#' * bar_len}  {freq:,}")


def parse_match_ids(raw: str | float | None) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def print_head(df: pd.DataFrame, n: int = HEAD_N) -> None:
    # Avoid truncating long addresses in the report
    with pd.option_context(
        "display.max_colwidth", 200,
        "display.width", 200,
        "display.max_columns", None,
        "display.expand_frame_repr", False,
    ):
        print(df.head(n).to_string(index=False))


def missing_report(df: pd.DataFrame, label: str) -> None:
    subbanner(f"Missing / empty values — {label}")
    print(f"  rows={len(df):,}  cols={list(df.columns)}")
    print(f"  {'column':<22} {'null':>12} {'blank(ws)':>12} {'null_or_blank':>14} {'pct_blank':>10}")
    n = len(df)
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        blanks = int(is_blank(df[col]).sum())
        pct = 100.0 * blanks / n if n else 0.0
        print(f"  {col:<22} {nulls:12,} {blanks:12,} {blanks:14,} {pct:9.3f}%")


def country_report(df: pd.DataFrame, label: str) -> None:
    subbanner(f"Country distribution — {label}")
    if "country" not in df.columns:
        print("  (no country column)")
        return
    vc = df["country"].fillna("(null)").value_counts(dropna=False)
    n = len(df)
    print(f"  unique country labels: {df['country'].nunique(dropna=False)}")
    print(f"  {'country':<20} {'count':>14} {'pct':>8}")
    for country, cnt in vc.items():
        print(f"  {str(country):<20} {cnt:14,} {100.0 * cnt / n:7.2f}%")


def id_integrity(df: pd.DataFrame, label: str, expected_prefix: str | None = None) -> None:
    subbanner(f"ID integrity — {label}")
    col = "entity_id" if "entity_id" in df.columns else "source1_entity_id"
    n = len(df)
    nunique = df[col].nunique(dropna=False)
    dups = n - nunique
    print(f"  rows={n:,}  unique {col}={nunique:,}  duplicate extra rows={dups:,}")
    if expected_prefix:
        bad = ~df[col].fillna("").str.startswith(expected_prefix)
        print(f"  IDs not starting with '{expected_prefix}': {int(bad.sum()):,}")


def text_stats(df: pd.DataFrame, label: str) -> None:
    subbanner(f"Text stats — {label}")
    for col in ("business_name", "business_address"):
        if col not in df.columns:
            continue
        blank = is_blank(df[col])
        lengths = df.loc[~blank, col].astype(str).str.len()
        print(f"  {col}:")
        print(f"    empty/null rate : {blank.mean() * 100:.3f}%  ({int(blank.sum()):,} / {len(df):,})")
        if lengths.empty:
            print("    (no non-empty values)")
            continue
        print(
            f"    length  mean={lengths.mean():.1f}  median={lengths.median():.0f}  "
            f"min={int(lengths.min())}  p95={lengths.quantile(0.95):.0f}  max={int(lengths.max())}"
        )
    if "country" in df.columns:
        print("  empty-address rate by country:")
        blank_addr = is_blank(df["business_address"])
        grp = (
            df.assign(_blank=blank_addr)
            .groupby(df["country"].fillna("(null)"), dropna=False)["_blank"]
            .agg(["mean", "sum", "size"])
        )
        for country, row in grp.iterrows():
            print(
                f"    {str(country):<16}  {row['mean'] * 100:6.2f}% empty  "
                f"({int(row['sum']):,} / {int(row['size']):,})"
            )


def lookup_record(by_id: dict[str, tuple], eid: str) -> tuple[str, str, str]:
    rec = by_id.get(eid)
    if rec is None:
        return ("<NOT FOUND>", "", "")
    return rec


def print_pair_block(s1_id: str, s1_rec: tuple, matches: list[tuple[str, tuple]]) -> None:
    s1_name, s1_addr, s1_cty = s1_rec
    print(f"\n  S1  {s1_id}  [{s1_cty}]")
    print(f"      NAME : {s1_name}")
    print(f"      ADDR : {s1_addr}")
    if not matches:
        print("      (no matches — singleton)")
        return
    for mid, rec in matches:
        name, addr, cty = rec
        print(f"    MATCH {mid}  [{cty}]")
        print(f"      NAME : {name}")
        print(f"      ADDR : {addr}")


def main() -> None:
    out_file = open(OUTPUT_PATH, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = Tee(orig_stdout, out_file)

    try:
        banner("Amazon ML Challenge 2026 — Phase 1 EDA")
        print(f"  started : {datetime.now().isoformat(timespec='seconds')}")
        print(f"  train   : {TRAIN_DIR}")
        print(f"  test    : {TEST_DIR}")
        print("  note    : all files read with sep='\\t' and dtype=str")

        # ------------------------------------------------------------------
        # Load once, reuse
        # ------------------------------------------------------------------
        banner("0. Load files (once)")
        train = {name: load_tsv(path) for name, path in TRAIN_FILES.items()}
        test = {name: load_tsv(path) for name, path in TEST_FILES.items()}

        all_frames = [
            ("train_source1", train["train_source1"], "S1-"),
            ("train_source2", train["train_source2"], "S2-"),
            ("train_source3", train["train_source3"], "S3-"),
            ("test_source1", test["test_source1"], "S1-"),
            ("test_source2", test["test_source2"], "S2-"),
            ("test_source3", test["test_source3"], "S3-"),
        ]

        # ------------------------------------------------------------------
        # 1. Shapes
        # ------------------------------------------------------------------
        banner("1. File shapes")
        print(f"  {'file':<24} {'rows':>14} {'cols':>6}  columns")
        for name, path in {**TRAIN_FILES, **TEST_FILES}.items():
            df = train[name] if name in train else test[name]
            print(f"  {name:<24} {df.shape[0]:14,} {df.shape[1]:6d}  {list(df.columns)}")

        # ------------------------------------------------------------------
        # 2. Sample rows
        # ------------------------------------------------------------------
        banner("2. Sample rows (head)")
        for name, df, _pfx in all_frames:
            subbanner(name)
            print_head(df)
        subbanner("train_ground_truth")
        print_head(train["train_ground_truth"])

        # ------------------------------------------------------------------
        # 3. Missing values
        # ------------------------------------------------------------------
        banner("3. Missing / empty values per column")
        for name, df, _pfx in all_frames:
            missing_report(df, name)
        missing_report(train["train_ground_truth"], "train_ground_truth")

        # ------------------------------------------------------------------
        # 4. Country distribution
        # ------------------------------------------------------------------
        banner("4. Country distribution (train vs test)")
        print(
            "  Training is expected to cover US + India only.\n"
            "  Test is expected to add France. Pipeline must treat country as an open set."
        )
        train_countries: set[str] = set()
        test_countries: set[str] = set()
        for name, df, _pfx in all_frames:
            country_report(df, name)
            labels = set(df["country"].dropna().unique())
            if name.startswith("train"):
                train_countries |= labels
            else:
                test_countries |= labels
        subbanner("Train vs test country set difference")
        print(f"  train countries : {sorted(train_countries)}")
        print(f"  test  countries : {sorted(test_countries)}")
        print(f"  test-only       : {sorted(test_countries - train_countries)}")
        print(f"  train-only      : {sorted(train_countries - test_countries)}")

        # ------------------------------------------------------------------
        # ID uniqueness (blocking-relevant)
        # ------------------------------------------------------------------
        banner("ID uniqueness / prefix checks")
        for name, df, pfx in all_frames:
            id_integrity(df, name, pfx)
        id_integrity(train["train_ground_truth"], "train_ground_truth", "S1-")

        # ------------------------------------------------------------------
        # 5. Ground-truth stats
        # ------------------------------------------------------------------
        banner("5. Ground-truth match statistics")
        gt = train["train_ground_truth"]
        s1 = train["train_source1"]

        match_lists = gt["matched_entity_ids"].map(parse_match_ids)
        n_matches = match_lists.map(len)
        n_s1_gt = len(gt)
        n_s1_src = len(s1)
        n_singletons = int((n_matches == 0).sum())
        n_non_single = n_s1_gt - n_singletons

        print(f"  GT rows                         : {n_s1_gt:,}")
        print(f"  Source 1 rows                   : {n_s1_src:,}")
        print(f"  GT covers all S1?               : {set(gt['source1_entity_id']) == set(s1['entity_id'])}")
        print(f"  singletons (0 matches)          : {n_singletons:,}  ({100.0 * n_singletons / n_s1_gt:.2f}%)")
        print(f"  non-singletons                  : {n_non_single:,}  ({100.0 * n_non_single / n_s1_gt:.2f}%)")
        if n_non_single:
            print(f"  avg matches / non-singleton     : {n_matches[n_matches > 0].mean():.4f}")
        print(f"  avg matches / all S1 entities   : {n_matches.mean():.4f}")
        print(f"  median matches (all)            : {n_matches.median():.0f}")
        print(f"  max matches on one S1 entity    : {int(n_matches.max())}")
        print(f"  entities with 10+ matches       : {int((n_matches >= 10).sum()):,}")
        print(f"  entities with 20+ matches       : {int((n_matches >= 20).sum()):,}")

        subbanner("Match-count histogram (count of S2/S3 IDs per S1 entity)")
        hist = n_matches.value_counts().sort_index()
        ascii_histogram(hist)
        print("\n  numeric value_counts:")
        for k, v in hist.items():
            print(f"    {int(k):>4} matches : {v:,} entities")

        # S2 vs S3 mix
        def source_mix(ids: list[str]) -> str:
            has2 = any(i.startswith("S2-") for i in ids)
            has3 = any(i.startswith("S3-") for i in ids)
            if not ids:
                return "none"
            if has2 and has3:
                return "S2+S3"
            if has2:
                return "S2_only"
            if has3:
                return "S3_only"
            return "other"

        mix = match_lists.map(source_mix)
        subbanner("Which sources appear in the match lists")
        mix_vc = mix.value_counts()
        for k, v in mix_vc.items():
            print(f"  {k:<12} {v:14,}  ({100.0 * v / n_s1_gt:.2f}%)")

        n_s2 = sum(sum(i.startswith("S2-") for i in ids) for ids in match_lists)
        n_s3 = sum(sum(i.startswith("S3-") for i in ids) for ids in match_lists)
        print(f"  total S2 IDs listed in GT : {n_s2:,}")
        print(f"  total S3 IDs listed in GT : {n_s3:,}")

        # Duplicate IDs inside a single GT list
        n_dup_in_list = int(sum(len(ids) != len(set(ids)) for ids in match_lists))
        print(f"  GT rows with duplicate IDs in the list : {n_dup_in_list:,}")

        # S2/S3 coverage: how many source records appear in GT at least once
        all_matched = [i for ids in match_lists for i in ids]
        matched_counter = Counter(all_matched)
        print(f"  unique S2/S3 IDs that appear in GT : {len(matched_counter):,}")
        multi_gt = sum(1 for _, c in matched_counter.items() if c > 1)
        print(f"  S2/S3 IDs linked to >1 S1 entity   : {multi_gt:,}  (should be 0 if 1:N from S1 only)")

        # ------------------------------------------------------------------
        # Country consistency S1 vs matches (blocking signal)
        # ------------------------------------------------------------------
        banner("Country consistency between S1 and its true matches")
        print("  If matches almost always share country, blocking can key on country")
        print("  without hard-coding the country *values*.")

        s1_country = dict(zip(s1["entity_id"], s1["country"]))
        s2_country = dict(zip(train["train_source2"]["entity_id"], train["train_source2"]["country"]))
        s3_country = dict(zip(train["train_source3"]["entity_id"], train["train_source3"]["country"]))
        id_to_country = {**s2_country, **s3_country}

        same = 0
        diff = 0
        missing_id = 0
        missing_cty = 0
        diff_examples = []
        for s1_id, ids in zip(gt["source1_entity_id"], match_lists):
            c1 = s1_country.get(s1_id)
            for mid in ids:
                c2 = id_to_country.get(mid)
                if c2 is None:
                    missing_id += 1
                    continue
                if pd.isna(c1) or pd.isna(c2) or str(c1).strip() == "" or str(c2).strip() == "":
                    missing_cty += 1
                    continue
                if str(c1) == str(c2):
                    same += 1
                else:
                    diff += 1
                    if len(diff_examples) < 8:
                        diff_examples.append((s1_id, c1, mid, c2))

        total_checked = same + diff
        print(f"  match-edges with same country     : {same:,}")
        print(f"  match-edges with DIFFERENT country: {diff:,}")
        print(f"  match IDs not found in S2/S3      : {missing_id:,}")
        print(f"  edges with missing country        : {missing_cty:,}")
        if total_checked:
            print(f"  same-country rate                 : {100.0 * same / total_checked:.4f}%")
        if diff_examples:
            print("  cross-country examples:")
            for a, ca, b, cb in diff_examples:
                print(f"    {a} [{ca}]  <->  {b} [{cb}]")

        # S1 country vs singleton rate
        subbanner("Singleton rate by S1 country")
        tmp = pd.DataFrame({
            "source1_entity_id": gt["source1_entity_id"],
            "n_matches": n_matches,
        })
        tmp["country"] = tmp["source1_entity_id"].map(s1_country)
        for cty, g in tmp.groupby("country", dropna=False):
            sing = int((g["n_matches"] == 0).sum())
            print(
                f"  {str(cty):<16}  n={len(g):,}  singletons={sing:,}  "
                f"({100.0 * sing / len(g):.2f}%)  avg_matches={g['n_matches'].mean():.3f}"
            )

        # ------------------------------------------------------------------
        # 6. Text stats
        # ------------------------------------------------------------------
        banner("6. Basic text stats (name / address length, empty rates)")
        for name, df, _pfx in all_frames:
            text_stats(df, name)

        # ------------------------------------------------------------------
        # 7. Side-by-side GT pairs
        # ------------------------------------------------------------------
        banner("7. Ground-truth matched pairs (side-by-side)")
        print(
            "  Sampled to cover: singletons, typical 1–4 match entities, high-match outliers,\n"
            "  and both train countries. Compare NAME/ADDR line-by-line for noise patterns."
        )

        s1_by_id = {
            eid: (name, addr, cty)
            for eid, name, addr, cty in zip(
                s1["entity_id"], s1["business_name"], s1["business_address"], s1["country"]
            )
        }
        s2s3_by_id = {
            eid: (name, addr, cty)
            for src in (train["train_source2"], train["train_source3"])
            for eid, name, addr, cty in zip(
                src["entity_id"], src["business_name"], src["business_address"], src["country"]
            )
        }

        gt_view = gt.copy()
        gt_view["n_matches"] = n_matches
        gt_view["country"] = gt_view["source1_entity_id"].map(s1_country)
        gt_view["match_list"] = match_lists

        rows: list[pd.Series] = []
        seen_ids: set[str] = set()

        def take(pool: pd.DataFrame, n: int, seed: int) -> None:
            if pool.empty:
                return
            n = min(n, len(pool))
            for _, row in pool.sample(n, random_state=seed).iterrows():
                sid = row["source1_entity_id"]
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    rows.append(row)

        for cty in sorted({c for c in gt_view["country"].dropna().unique()}):
            take(gt_view[(gt_view["n_matches"] == 0) & (gt_view["country"] == cty)], 1, RNG_SEED)
        for k in (1, 2, 3, 4, 6):
            for cty in sorted({c for c in gt_view["country"].dropna().unique()}):
                take(
                    gt_view[(gt_view["n_matches"] == k) & (gt_view["country"] == cty)],
                    1,
                    RNG_SEED + k,
                )
        take(gt_view[gt_view["n_matches"] >= 10], 3, RNG_SEED + 99)
        # a few extra random non-singletons for variety
        take(gt_view[gt_view["n_matches"] > 0], 4, RNG_SEED + 7)

        print(f"  showing {len(rows)} S1 entities")
        for row in rows:
            s1_id = row["source1_entity_id"]
            ids = row["match_list"]
            # For high-match entities, print first 6 matches to keep the report readable
            shown_ids = ids if len(ids) <= 6 else ids[:6]
            match_recs = [(mid, lookup_record(s2s3_by_id, mid)) for mid in shown_ids]
            print_pair_block(s1_id, lookup_record(s1_by_id, s1_id), match_recs)
            if len(ids) > 6:
                print(f"      ... {len(ids) - 6} further matches omitted (total {len(ids)})")

        # ------------------------------------------------------------------
        # Blocking-scale reminder
        # ------------------------------------------------------------------
        banner("Scale reminder for Phase 3 (blocking)")
        n1 = len(train["train_source1"])
        n2 = len(train["train_source2"])
        n3 = len(train["train_source3"])
        brute = n1 * (n2 + n3)
        print(f"  train S1 × (S2+S3) brute-force pairs : {brute:,.0f}")
        print("  Brute force is not feasible. Blocking must cut this by orders of magnitude")
        print("  while keeping recall high (missed true pairs can never be recovered).")
        t1, t2, t3 = len(test["test_source1"]), len(test["test_source2"]), len(test["test_source3"])
        print(f"  test  S1 × (S2+S3) brute-force pairs : {t1 * (t2 + t3):,.0f}")

        banner("EDA complete")
        print(f"  finished : {datetime.now().isoformat(timespec='seconds')}")
        print(f"  writing  : {OUTPUT_PATH}")

    finally:
        sys.stdout = orig_stdout
        out_file.close()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
