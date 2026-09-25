import pandas as pd, sys, os
sys.path.insert(0, os.path.join("code","business_entity_resolution","src"))
sys.path.insert(0, os.path.join("code","business_entity_resolution"))
from blocking.ensemble import ensemble_block
from features.pair_features import feature_pair
import os, random

def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8").fillna("")

print("Loading test...")
s1_test = load_tsv("dataset/test/test_source1.tsv")
s2_test = load_tsv("dataset/test/test_source2.tsv")
s3_test = load_tsv("dataset/test/test_source3.tsv")
print(f"test s1 {len(s1_test)}")

# Process per country in batches to avoid long single loop
from collections import defaultdict
import time

# Build pool per country once
pool = pd.concat([s2_test, s3_test], ignore_index=True)

# Heuristic thresholds tuned for F0.5 precision-heavy: require high name similarity
NAME_THR = 0.88
ADDR_THR = 0.3  # low addr allowed if name very high
HIGH_NAME_THR = 0.95

from blocking.ensemble import build_index, candidates_for_s1

def predict_for_country(s1_c, s2_c, s3_c):
    pool_c = pd.concat([s2_c, s3_c], ignore_index=True)
    print(f"  building index for pool {len(pool_c)}")
    import time
    t0=time.time()
    inv, postal_inv, pool_df, id_to_row = build_index(pool_c)
    print(f"  index built {time.time()-t0:.1f}s, tokens {len(inv)}")
    # map for features
    pmap = id_to_row
    pred={}
    cands={}
    t1=time.time()
    for _, s1_row in s1_c.iterrows():
        clist = candidates_for_s1(s1_row, inv, postal_inv, pmap, max_candidates=15)
        cands[s1_row["entity_id"]]=clist
        chosen=[]
        for pid in clist:
            prow = pmap.get(pid)
            if prow is None:
                continue
            feats = feature_pair(s1_row, prow)
            ns = feats["name_token_set"]
            asn = feats["name_token_set_nosuf"]
            if ns >= HIGH_NAME_THR:
                chosen.append(pid)
            elif ns >= NAME_THR and feats["addr_token_set"] >= ADDR_THR:
                chosen.append(pid)
            elif asn >= 0.92 and feats["name_jaccard_nosuf"]>=0.7:
                chosen.append(pid)
        if chosen:
            max_ns = max([feature_pair(s1_row, pmap[pid])["name_token_set"] for pid in chosen])
            if max_ns < 0.82:
                chosen=[]
        pred[s1_row["entity_id"]]=chosen
    print(f"  scoring done {time.time()-t1:.1f}s for {len(s1_c)} S1")
    return pred, cands

all_pred={}
all_cands={}
for country in s1_test["country"].unique():
    print(f"Country {country}")
    s1_c = s1_test[s1_test["country"]==country]
    s2_c = s2_test[s2_test["country"]==country]
    s3_c = s3_test[s3_test["country"]==country]
    pred, cands = predict_for_country(s1_c, s2_c, s3_c)
    all_pred.update(pred)
    all_cands.update(cands)

# ensure all S1 have entry
for sid in s1_test["entity_id"]:
    all_pred.setdefault(sid, [])
    all_cands.setdefault(sid, [])

# write
os.makedirs("output", exist_ok=True)
with open("output/matching_results.tsv","w",encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_test["entity_id"]:
        f.write(f"{sid}\t{','.join(all_pred[sid])}\n")
with open("output/candidate_pairs.tsv","w",encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_test["entity_id"]:
        f.write(f"{sid}\t{','.join(all_cands[sid])}\n")
print("done")
