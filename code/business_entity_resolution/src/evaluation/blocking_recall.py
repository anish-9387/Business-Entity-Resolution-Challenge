import numpy as np

def evaluate_blocking(candidates, gt_dict, s1_ids):
    """
    Evaluate blocking quality.
    
    candidates: dict[s1_id -> list[pool_id]]
    gt_dict: dict[s1_id -> set[matched_ids]]
    s1_ids: list of s1_ids to evaluate
    
    Returns: dict with metrics
    """
    total_true = 0
    found_true = 0
    candidate_counts = []
    
    for sid in s1_ids:
        true_matches = gt_dict.get(sid, set())
        cand_set = set(candidates.get(sid, []))
        candidate_counts.append(len(cand_set))
        
        for mid in true_matches:
            total_true += 1
            if mid in cand_set:
                found_true += 1
    
    counts = np.array(candidate_counts)
    # Total pool size (approximate - sum of all candidate sets' unique IDs)
    all_pool = set()
    for cands in candidates.values():
        all_pool.update(cands)
    total_pool = max(len(all_pool), 1)
    
    metrics = {
        'candidate_recall': found_true / max(total_true, 1),
        'total_true_matches': total_true,
        'found_true_matches': found_true,
        'missed_true_matches': total_true - found_true,
        'avg_candidates': float(counts.mean()) if len(counts) > 0 else 0,
        'median_candidates': float(np.median(counts)) if len(counts) > 0 else 0,
        'p95_candidates': float(np.percentile(counts, 95)) if len(counts) > 0 else 0,
        'max_candidates': int(counts.max()) if len(counts) > 0 else 0,
        'min_candidates': int(counts.min()) if len(counts) > 0 else 0,
        'zero_candidate_count': int((counts == 0).sum()),
        'total_s1': len(s1_ids),
        'reduction_ratio': 1.0 - (counts.sum() / (len(s1_ids) * total_pool)) if total_pool > 0 else 0,
    }
    
    print(f"  Blocking recall: {metrics['candidate_recall']:.4f} ({found_true}/{total_true})")
    print(f"  Candidates: avg={metrics['avg_candidates']:.1f}, median={metrics['median_candidates']:.0f}, p95={metrics['p95_candidates']:.0f}, max={metrics['max_candidates']}")
    print(f"  Zero-candidate S1 entities: {metrics['zero_candidate_count']}/{metrics['total_s1']}")
    
    return metrics
