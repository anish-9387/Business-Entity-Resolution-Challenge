def analyze_errors(pred, gt, candidates=None):
    """
    Generate false positive and false negative reports.
    
    pred: dict[s1_id -> set/list[pool_id]]
    gt: dict[s1_id -> set[pool_id]]
    candidates: dict[s1_id -> list[pool_id]] (optional, for FN analysis)
    
    Returns: (fp_report, fn_report)
    fp_report: list of dicts with keys: s1_id, wrong_id, type
    fn_report: list of dicts with keys: s1_id, missed_id, was_candidate, type
    """
    fp_report = []
    fn_report = []
    
    all_s1 = set(gt.keys()) | set(pred.keys())
    
    for sid in all_s1:
        true_set = set(gt.get(sid, set()))
        pred_set = set(pred.get(sid, []))
        cand_set = set(candidates.get(sid, [])) if candidates else None
        
        # False positives: predicted but not true
        for pid in pred_set - true_set:
            fp_type = 'singleton_violation' if not true_set else 'wrong_match'
            fp_report.append({'s1_id': sid, 'wrong_id': pid, 'type': fp_type})
        
        # False negatives: true but not predicted
        for mid in true_set - pred_set:
            was_cand = mid in cand_set if cand_set is not None else None
            fn_type = 'blocking_miss' if was_cand is False else ('matching_miss' if was_cand else 'unknown')
            fn_report.append({'s1_id': sid, 'missed_id': mid, 'was_candidate': was_cand, 'type': fn_type})
    
    # Summary stats
    n_fp = len(fp_report)
    n_fn = len(fn_report)
    n_singleton_viol = sum(1 for r in fp_report if r['type'] == 'singleton_violation')
    n_blocking_miss = sum(1 for r in fn_report if r['type'] == 'blocking_miss')
    n_matching_miss = sum(1 for r in fn_report if r['type'] == 'matching_miss')
    
    print(f"  Error Analysis:")
    print(f"    False Positives: {n_fp} (singleton violations: {n_singleton_viol})")
    print(f"    False Negatives: {n_fn} (blocking misses: {n_blocking_miss}, matching misses: {n_matching_miss})")
    
    return fp_report, fn_report
