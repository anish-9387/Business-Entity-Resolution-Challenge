import numpy as np
from src.evaluation.f05 import macro_f05

def tune_f05(val_s1_scores, gt_dict, val_ids):
    """
    Tune threshold and margin for best macro F0.5.
    
    val_s1_scores: dict[s1_id -> list[(pool_id, score)]]
    gt_dict: dict[s1_id -> set[matched_ids]]
    val_ids: list of s1_ids to evaluate on
    
    Returns: (best_threshold, best_margin, best_f05_score)
    """
    best_thr, best_margin, best_f05 = 0.5, 0.0, -1.0
    
    # Phase 1: coarse threshold search (no margin)
    for thr in np.arange(0.30, 0.96, 0.05):
        pred = {}
        for sid in val_ids:
            lst = val_s1_scores.get(sid, [])
            pred[sid] = set(pid for pid, s in lst if s >= thr)
        f = macro_f05(gt_dict, pred, list(val_ids))
        if f > best_f05:
            best_f05 = f
            best_thr = thr
    
    # Phase 2: fine threshold refinement
    for thr in np.arange(max(0.1, best_thr - 0.06), min(1.0, best_thr + 0.07), 0.01):
        pred = {}
        for sid in val_ids:
            lst = val_s1_scores.get(sid, [])
            pred[sid] = set(pid for pid, s in lst if s >= thr)
        f = macro_f05(gt_dict, pred, list(val_ids))
        if f > best_f05:
            best_f05 = f
            best_thr = thr
    
    # Phase 3: margin search at best threshold
    # margin rule: only accept candidates whose score is within margin of top score
    # Or: require top/second gap >= margin for ambiguous cases
    best_margin = 0.0
    for margin in np.arange(0.0, 0.25, 0.02):
        pred = {}
        for sid in val_ids:
            lst = val_s1_scores.get(sid, [])
            if not lst:
                pred[sid] = set()
                continue
            lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
            top_score = lst_sorted[0][1]
            if top_score < best_thr:
                pred[sid] = set()
                continue
            # Accept all above threshold, but if ambiguous (top-second < margin), be more conservative
            second_score = lst_sorted[1][1] if len(lst_sorted) > 1 else 0.0
            chosen = set()
            for pid, s in lst_sorted:
                if s >= best_thr:
                    # If this isn't the top and the gap to top is small, require higher threshold
                    if s < top_score and (top_score - s) < margin:
                        # ambiguous: only accept if score itself is very high
                        if s >= best_thr + 0.05:
                            chosen.add(pid)
                    else:
                        chosen.add(pid)
            pred[sid] = chosen
        f = macro_f05(gt_dict, pred, list(val_ids))
        if f > best_f05:
            best_f05 = f
            best_margin = margin
    
    print(f"  Tuned: threshold={best_thr:.3f}, margin={best_margin:.3f}, F0.5={best_f05:.4f}")
    return best_thr, best_margin, best_f05
