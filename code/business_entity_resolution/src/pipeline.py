import pandas as pd
import os, random
from collections import defaultdict
from src.blocking.ensemble import ensemble_block
from src.features.pair_features import feature_pair
from src.evaluation.f05 import macro_f05, parse_gt, entity_f05
import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB=True
except:
    HAS_LGB=False

def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8").fillna("")

def build_features_for_candidates(s1_df, s2_df, s3_df, candidates, gt_dict=None):
    # map id->row
    s1_map={r["entity_id"]:r for _,r in s1_df.iterrows()}
    s2_map={r["entity_id"]:r for _,r in s2_df.iterrows()}
    s3_map={r["entity_id"]:r for _,r in s3_df.iterrows()}
    pool = {**s2_map, **s3_map}
    rows=[]
    for sid, cands in candidates.items():
        s1 = s1_map.get(sid)
        if s1 is None:
            continue
        for pid in cands:
            prow = pool.get(pid)
            if prow is None:
                continue
            feats = feature_pair(s1, prow)
            label = 1 if gt_dict is not None and pid in gt_dict.get(sid, set()) else 0
            # need negative label even if gt_dict is None? then label 0
            rows.append((sid, pid, feats, label))
    return rows

def train_and_tune(sample_s1_ids, s1_df, s2_df, s3_df, gt_df, val_ratio=0.2):
    # split by S1 entity
    random.seed(42)
    ids = list(sample_s1_ids)
    random.shuffle(ids)
    n_val = int(len(ids)*val_ratio)
    train_ids = set(ids[n_val:])
    val_ids = set(ids[:n_val])

    # block for train+val together (candidates)
    # we filter s1_df to sample
    s1_sample = s1_df[s1_df["entity_id"].isin(ids)]
    candidates = ensemble_block(s1_sample, s2_df, s3_df, max_candidates=20)
    gt_dict = parse_gt(gt_df)

    rows = build_features_for_candidates(s1_sample, s2_df, s3_df, candidates, gt_dict)
    # split rows
    train_rows = [r for r in rows if r[0] in train_ids]
    val_rows = [r for r in rows if r[0] in val_ids]

    # feature cols
    cols = list(train_rows[0][2].keys()) if train_rows else []
    import pandas as pd
    X_train = pd.DataFrame([r[2] for r in train_rows])
    y_train = np.array([r[3] for r in train_rows])
    X_val = pd.DataFrame([r[2] for r in val_rows])
    y_val = np.array([r[3] for r in val_rows])

    # handle imbalance
    print(f"train pairs {len(train_rows)} pos {y_train.sum()} neg {(y_train==0).sum()}")
    print(f"val pairs {len(val_rows)} pos {y_val.sum()}")

    if HAS_LGB and len(train_rows)>1000:
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        params = {"objective":"binary","metric":"auc","verbosity":-1,"boosting_type":"gbdt","learning_rate":0.05,"num_leaves":31}
        model = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dtrain,dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    else:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=500, class_weight="balanced")
        model.fit(X_train, y_train)

    # predict val
    if HAS_LGB:
        val_pred = model.predict(X_val)
    else:
        val_pred = model.predict_proba(X_val)[:,1]

    # need to map val scores per S1
    from collections import defaultdict
    # group val_rows scores
    val_scores = defaultdict(list)
    for (sid,pid,_,_), score in zip(val_rows, val_pred):
        val_scores[sid].append((pid, score))

    # tune threshold for macro F0.5
    best_thr, best_f = 0.5, -1
    gt_sub = {k:gt_dict[k] for k in val_ids}
    for thr in np.arange(0.3, 0.95, 0.05):
        pred = {}
        for sid in val_ids:
            lst = val_scores.get(sid, [])
            pred[sid] = set([pid for pid,s in lst if s>=thr])
        f = macro_f05(gt_sub, pred, list(val_ids))
        if f>best_f:
            best_f=f
            best_thr=thr
    print(f"best thr {best_thr:.2f} f05 {best_f:.4f}")
    # refine
    for thr in np.arange(best_thr-0.04, best_thr+0.05, 0.01):
        if thr<0 or thr>1: continue
        pred={}
        for sid in val_ids:
            lst=val_scores.get(sid,[])
            pred[sid]=set([pid for pid,s in lst if s>=thr])
        f=macro_f05(gt_sub,pred,list(val_ids))
        if f>best_f:
            best_f=f
            best_thr=thr
    print(f"refined thr {best_thr:.3f} f05 {best_f:.4f}")
    return model, cols, best_thr

def predict_test(s1_df, s2_df, s3_df, model, cols, thr):
    candidates = ensemble_block(s1_df, s2_df, s3_df, max_candidates=20)
    rows = build_features_for_candidates(s1_df, s2_df, s3_df, candidates, gt_dict=None)
    if not rows:
        return {}, candidates
    import pandas as pd
    # need to keep mapping
    # build X
    X = pd.DataFrame([r[2] for r in rows])
    # ensure cols order
    X = X[cols]
    if HAS_LGB:
        scores = model.predict(X)
    else:
        scores = model.predict_proba(X)[:,1]
    from collections import defaultdict
    s1_scores = defaultdict(list)
    for (sid,pid,_,_), s in zip(rows, scores):
        s1_scores[sid].append((pid,s))
    pred={}
    for sid in s1_df["entity_id"]:
        lst=s1_scores.get(sid, [])
        # sort by score desc
        lst_sorted=sorted(lst, key=lambda x: x[1], reverse=True)
        # apply thr
        chosen=[pid for pid,s in lst_sorted if s>=thr]
        # margin rule: if top score < thr, keep empty (singleton)
        # already done
        pred[sid]=chosen
    return pred, candidates

def write_outputs(pred, candidates, s1_ids, match_path, cand_path):
    os.makedirs(os.path.dirname(match_path), exist_ok=True)
    with open(match_path,"w",encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            mids = pred.get(sid, [])
            f.write(f"{sid}\t{','.join(mids)}\n")
    with open(cand_path,"w",encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            cands = candidates.get(sid, [])
            f.write(f"{sid}\t{','.join(cands)}\n")
