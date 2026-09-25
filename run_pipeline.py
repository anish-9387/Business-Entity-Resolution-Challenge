import pandas as pd
import sys
sys.path.insert(0, "code/business_entity_resolution/src")
sys.path.insert(0, "code/business_entity_resolution")
from src.pipeline import load_tsv, train_and_tune, predict_test, write_outputs
from src.config import *
import random

print("Loading train...")
s1_train = load_tsv(TRAIN_S1)
s2_train = load_tsv(TRAIN_S2)
s3_train = load_tsv(TRAIN_S3)
gt_train = load_tsv(TRAIN_GT)
print(f"train s1 {len(s1_train)} s2 {len(s2_train)} s3 {len(s3_train)}")

print("Loading test...")
s1_test = load_tsv(TEST_S1)
s2_test = load_tsv(TEST_S2)
s3_test = load_tsv(TEST_S3)
print(f"test s1 {len(s1_test)} s2 {len(s2_test)} s3 {len(s3_test)}")

# sample 5000 S1 for training (to keep runtime reasonable)
sample_n = 5000
if len(s1_train) > sample_n:
    random.seed(42)
    sample_ids = random.sample(s1_train["entity_id"].tolist(), sample_n)
else:
    sample_ids = s1_train["entity_id"].tolist()
# subsample pool for training to keep blocking fast
# sample 200k from s2/s3 per country
def subsample_pool(df, n=200000):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)
s2_train_small = pd.concat([subsample_pool(s2_train[s2_train["country"]==c], 100000) for c in s2_train["country"].unique()])
s3_train_small = pd.concat([subsample_pool(s3_train[s3_train["country"]==c], 100000) for c in s3_train["country"].unique()])


print(f"Training on {len(sample_ids)} S1 entities...")
# use subsampled pool for training
model, cols, thr = train_and_tune(sample_ids, s1_train, s2_train_small, s3_train_small, gt_train, val_ratio=0.2)

print("Predicting test...")
pred, candidates = predict_test(s1_test, s2_test, s3_test, model, cols, thr)

# ensure order is test S1 order
s1_ids = s1_test["entity_id"].tolist()
write_outputs(pred, candidates, s1_ids, OUTPUT_MATCH, OUTPUT_CAND)
print(f"Done. Wrote {OUTPUT_MATCH} and {OUTPUT_CAND}")
# quick stats
nonempty = sum(1 for v in pred.values() if v)
print(f"non-empty preds: {nonempty}/{len(pred)} avg candidates {sum(len(v) for v in candidates.values())/len(candidates):.2f}")
