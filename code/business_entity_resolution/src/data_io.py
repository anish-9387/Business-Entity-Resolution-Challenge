import pandas as pd
import os

def load_tsv(path: str) -> pd.DataFrame:
    """Load a TSV file with explicit tab separator."""
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8").fillna("")

def write_matching_results(pred: dict, s1_ids: list, path: str):
    """Write matching_results.tsv in competition format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            mids = pred.get(sid, [])
            f.write(f"{sid}\t{','.join(mids)}\n")

def write_candidate_pairs(candidates: dict, s1_ids: list, path: str):
    """Write candidate_pairs.tsv in competition format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            cands = candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(cands)}\n")
