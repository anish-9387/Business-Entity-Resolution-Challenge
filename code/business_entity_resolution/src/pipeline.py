"""
Full pipeline: Ingest -> Normalize -> Block -> Features -> Train -> Calibrate -> Tune -> Predict -> Output
Implements all stages from PLAN.md §52 end-to-end pseudocode.
"""
import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

from src.data_io import load_tsv, write_matching_results, write_candidate_pairs
from src.blocking.ensemble import ensemble_block
from src.features.pair_features import compute_all_features
from src.evaluation.f05 import macro_f05, parse_gt
from src.evaluation.blocking_recall import evaluate_blocking
from src.evaluation.error_analysis import analyze_errors
from src.model.train import train_gbm
from src.model.calibrate import fit_calibrator, calibrate
from src.model.predict import predict_scores, predict_calibrated
from src.decision.threshold import tune_f05
from src.decision.singleton import apply_singleton_filter
from src.decision.consistency import resolve_cross_source
from src.config import RANDOM_SEED

def build_features_for_candidates(s1_df, s2_df, s3_df, candidates, channel_evidence=None, gt_dict=None):
    s1_map = {r["entity_id"]: r for _, r in s1_df.iterrows()}
    pool_map = {}
    for _, r in s2_df.iterrows(): pool_map[r["entity_id"]] = r
    for _, r in s3_df.iterrows(): pool_map[r["entity_id"]] = r
    rows = []
    for sid, cands in candidates.items():
        s1 = s1_map.get(sid)
        if s1 is None: continue
        for pid in cands:
            prow = pool_map.get(pid)
            if prow is None: continue
            feats = compute_all_features(s1, prow, channel_evidence=channel_evidence, s1_id=sid, pool_id=pid)
            label = 1 if gt_dict is not None and pid in gt_dict.get(sid, set()) else 0
            rows.append((sid, pid, feats, label))
    return rows

def add_hard_negatives(rows, s1_df, s2_df, s3_df, gt_dict, max_extra=2000):
    from src.normalization import normalize_text
    from rapidfuzz import fuzz
    existing = {(sid, pid) for sid, pid, _, _ in rows}
    pool_map = {}
    for _, r in s2_df.iterrows(): pool_map[r["entity_id"]] = r
    for _, r in s3_df.iterrows(): pool_map[r["entity_id"]] = r
    s1_map = {r["entity_id"]: r for _, r in s1_df.iterrows()}
    hard_negs = []
    pool_names = {pid: normalize_text(r.get("business_name", "")) for pid, r in pool_map.items()}
    for sid, true_ids in gt_dict.items():
        if not true_ids or sid not in s1_map: continue
        s1_row = s1_map[sid]
        s1_name = normalize_text(s1_row.get("business_name", ""))
        if not s1_name: continue
        count = 0
        for pid in true_ids:
            if pid not in pool_map: continue
            for other_pid, other_name in pool_names.items():
                if other_pid in true_ids or (sid, other_pid) in existing: continue
                if fuzz.token_set_ratio(s1_name, other_name) >= 70:
                    feats = compute_all_features(s1_row, pool_map[other_pid], None, sid, other_pid)
                    hard_negs.append((sid, other_pid, feats, 0))
                    existing.add((sid, other_pid))
                    count += 1
                    if count >= 5: break
            if count >= 5: break
        if len(hard_negs) >= max_extra: break
    print(f"  Added {len(hard_negs)} hard negatives")
    return rows + hard_negs

def train_pipeline(s1_df, s2_df, s3_df, gt_df, val_ratio=0.2):
    random.seed(RANDOM_SEED)
    gt_dict = parse_gt(gt_df)
    all_ids = s1_df["entity_id"].tolist()
    random.shuffle(all_ids)
    n_val = int(len(all_ids) * val_ratio)
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val S1 entities")
    print("Blocking...")
    candidates, channel_evidence = ensemble_block(s1_df, s2_df, s3_df, max_candidates=50)
    print("Evaluating blocking recall...")
    evaluate_blocking(candidates, gt_dict, all_ids)
    print("Building features...")
    rows = build_features_for_candidates(s1_df, s2_df, s3_df, candidates, channel_evidence, gt_dict)
    print("Mining hard negatives...")
    rows = add_hard_negatives(rows, s1_df, s2_df, s3_df, gt_dict, 3000)
    train_rows = [r for r in rows if r[0] in train_ids]
    val_rows = [r for r in rows if r[0] in val_ids]
    if not train_rows: raise ValueError("No training rows generated.")
    feature_cols = sorted(train_rows[0][2].keys())
    X_train = pd.DataFrame([r[2] for r in train_rows])[feature_cols]
    y_train = np.array([r[3] for r in train_rows])
    X_val = pd.DataFrame([r[2] for r in val_rows])[feature_cols] if val_rows else pd.DataFrame()
    y_val = np.array([r[3] for r in val_rows]) if val_rows else np.array([])
    print(f"Train: {len(train_rows)} pairs ({y_train.sum():.0f} pos, {(y_train==0).sum():.0f} neg)")
    print(f"Val:   {len(val_rows)} pairs ({y_val.sum():.0f} pos)")
    print("Training model...")
    model = train_gbm(X_train, y_train, X_val, y_val)
    print("Predicting validation...")
    val_pred = predict_scores(model, X_val) if len(X_val) > 0 else np.array([])
    calibrator = None
    if len(val_pred) > 100:
        print("Fitting calibrator...")
        calibrator = fit_calibrator(y_val, val_pred)
        val_pred = calibrate(calibrator, val_pred)
    val_s1_scores = defaultdict(list)
    for (sid, pid, _, _), score in zip(val_rows, val_pred):
        val_s1_scores[sid].append((pid, float(score)))
    print("Tuning threshold...")
    gt_val = {k: gt_dict.get(k, set()) for k in val_ids}
    threshold, margin, best_f05 = tune_f05(val_s1_scores, gt_val, list(val_ids))
    print("Error analysis...")
    val_pred_dict = {}
    for sid in val_ids:
        val_pred_dict[sid] = apply_singleton_filter(sid, val_s1_scores.get(sid, []), threshold, margin)
    analyze_errors(val_pred_dict, gt_val, candidates)
    print(f"\nTraining complete: threshold={threshold:.3f}, margin={margin:.3f}, F0.5={best_f05:.4f}")
    return model, calibrator, threshold, margin, feature_cols

def predict_pipeline(s1_df, s2_df, s3_df, model, calibrator, threshold, margin, feature_cols):
    print("Blocking test set...")
    candidates, channel_evidence = ensemble_block(s1_df, s2_df, s3_df, max_candidates=50)
    print("Building test features...")
    rows = build_features_for_candidates(s1_df, s2_df, s3_df, candidates, channel_evidence, None)
    if not rows: return {sid: [] for sid in s1_df["entity_id"]}, candidates
    X = pd.DataFrame([r[2] for r in rows])[feature_cols]
    print("Scoring...")
    scores = predict_calibrated(model, X, calibrator)
    s1_scores = defaultdict(list)
    for (sid, pid, _, _), s in zip(rows, scores):
        s1_scores[sid].append((pid, float(s)))
    print("Applying decision policy...")
    pred_scored = {}
    for sid in s1_df["entity_id"]:
        cands_with_scores = s1_scores.get(sid, [])
        matched = apply_singleton_filter(sid, cands_with_scores, threshold, margin)
        pred_scored[sid] = [(pid, score) for pid, score in cands_with_scores if pid in set(matched)]
    print("Cross-source consistency...")
    pred = resolve_cross_source(pred_scored)
    for sid in s1_df["entity_id"]: pred.setdefault(sid, [])
    return pred, candidates
