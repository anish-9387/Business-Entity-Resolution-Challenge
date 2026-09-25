"""
run_fast.py — Quick heuristic-only submission (no model training)
"""
import sys
import os
import time

sys.path.insert(0, os.path.join("code", "business_entity_resolution"))

from src.data_io import load_tsv, write_matching_results, write_candidate_pairs
from src.blocking.ensemble import ensemble_block
from src.features.pair_features import compute_all_features
from src.config import TEST_S1, TEST_S2, TEST_S3, OUTPUT_MATCH, OUTPUT_CAND

print("Loading test data...")
s1_test = load_tsv(TEST_S1)
s2_test = load_tsv(TEST_S2)
s3_test = load_tsv(TEST_S3)
print(f"  S1: {len(s1_test)}, S2: {len(s2_test)}, S3: {len(s3_test)}")

t0 = time.time()
print("Running blocking...")
candidates, channel_evidence = ensemble_block(s1_test, s2_test, s3_test, max_candidates=20)
print(f"Blocking done in {time.time() - t0:.1f}s")

pool_map = {}
for df in [s2_test, s3_test]:
    for _, r in df.iterrows(): pool_map[r["entity_id"]] = r
s1_map = {r["entity_id"]: r for _, r in s1_test.iterrows()}

NAME_THR = 0.88
HIGH_NAME_THR = 0.95
ADDR_THR = 0.30
NOSUF_THR = 0.92

print("Scoring candidates...")
t1 = time.time()
pred = {}
for sid in s1_test["entity_id"]:
    s1_row = s1_map[sid]
    cands = candidates.get(sid, [])
    chosen = []
    best_name_score = 0.0

    for pid in cands:
        prow = pool_map.get(pid)
        if prow is None: continue

        feats = compute_all_features(s1_row, prow, channel_evidence=channel_evidence, s1_id=sid, pool_id=pid)
        ns = feats.get("name_token_set", 0)
        ns_nosuf = feats.get("name_token_set_nosuf", 0)
        addr_sim = feats.get("addr_token_set", 0)
        name_jac_nosuf = feats.get("name_jaccard_nosuf", 0)
        channels = feats.get("blocking_channel_count", 0)

        best_name_score = max(best_name_score, ns)

        if ns >= HIGH_NAME_THR: chosen.append(pid)
        elif ns >= NAME_THR and addr_sim >= ADDR_THR: chosen.append(pid)
        elif ns_nosuf >= NOSUF_THR and name_jac_nosuf >= 0.7: chosen.append(pid)
        elif channels >= 3 and ns >= 0.82 and addr_sim >= 0.25: chosen.append(pid)

    if chosen and best_name_score < 0.80: chosen = []
    pred[sid] = chosen

print(f"Scoring done in {time.time() - t1:.1f}s")
for sid in s1_test["entity_id"]: pred.setdefault(sid, [])

s1_ids = s1_test["entity_id"].tolist()
write_matching_results(pred, s1_ids, OUTPUT_MATCH)
write_candidate_pairs(candidates, s1_ids, OUTPUT_CAND)
print(f"\nDone!")
