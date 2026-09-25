"""
run_pipeline.py — Full end-to-end training + test prediction
"""
import sys
import os
import random

sys.path.insert(0, os.path.join("code", "business_entity_resolution"))

from src.data_io import load_tsv, write_matching_results, write_candidate_pairs
from src.config import *
from src.pipeline import train_pipeline, predict_pipeline

print("=" * 60)
print("Business Entity Resolution — Full Pipeline")
print("=" * 60)

print("\nLoading training data...")
s1_train = load_tsv(TRAIN_S1)
s2_train = load_tsv(TRAIN_S2)
s3_train = load_tsv(TRAIN_S3)
gt_train = load_tsv(TRAIN_GT)
print(f"  S1: {len(s1_train)}, S2: {len(s2_train)}, S3: {len(s3_train)}")

print("Loading test data...")
s1_test = load_tsv(TEST_S1)
s2_test = load_tsv(TEST_S2)
s3_test = load_tsv(TEST_S3)
print(f"  S1: {len(s1_test)}, S2: {len(s2_test)}, S3: {len(s3_test)}")

random.seed(RANDOM_SEED)
if len(s1_train) > SAMPLE_N:
    sample_ids = set(random.sample(s1_train["entity_id"].tolist(), SAMPLE_N))
    s1_train_sub = s1_train[s1_train["entity_id"].isin(sample_ids)].copy()
else:
    s1_train_sub = s1_train.copy()

def subsample_pool(df, n=200000):
    if len(df) <= n: return df
    return df.sample(n=n, random_state=RANDOM_SEED)

s2_sub = subsample_pool(s2_train, 200000)
s3_sub = subsample_pool(s3_train, 200000)
print(f"Training subsample: {len(s1_train_sub)} S1, {len(s2_sub)} S2, {len(s3_sub)} S3")

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)
model, calibrator, threshold, margin, feature_cols = train_pipeline(
    s1_train_sub, s2_sub, s3_sub, gt_train, val_ratio=0.2
)

print("\n" + "=" * 60)
print("TEST PREDICTION")
print("=" * 60)
pred, candidates = predict_pipeline(
    s1_test, s2_test, s3_test,
    model, calibrator, threshold, margin, feature_cols
)

s1_ids = s1_test["entity_id"].tolist()
write_matching_results(pred, s1_ids, OUTPUT_MATCH)
write_candidate_pairs(candidates, s1_ids, OUTPUT_CAND)

nonempty = sum(1 for v in pred.values() if v)
total_matched = sum(len(v) for v in pred.values())
avg_cands = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
print(f"\nDone!")
print(f"  Wrote: {OUTPUT_MATCH}")
print(f"  Wrote: {OUTPUT_CAND}")
print(f"  Non-empty predictions: {nonempty}/{len(pred)}")
print(f"  Total matched IDs: {total_matched}")
print(f"  Avg candidates/S1: {avg_cands:.1f}")
