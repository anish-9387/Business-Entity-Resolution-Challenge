from collections import defaultdict

def resolve_cross_source(pred_dict, mode='keep_strongest'):
    """
    Cross-source consistency: if the same S2/S3 record is claimed by multiple S1 entities,
    keep only the strongest claim (highest score assignment).
    
    pred_dict: dict[s1_id -> list[(pool_id, score)]] or dict[s1_id -> list[pool_id]]
    mode: 'keep_strongest' (default) - keep only highest scoring S1 claim
          'keep_all' - no filtering
    
    Returns: dict[s1_id -> list[pool_id]] (filtered)
    """
    if mode == 'keep_all':
        # Just extract pool_ids if scored
        result = {}
        for sid, v in pred_dict.items():
            if v and isinstance(v[0], tuple):
                result[sid] = [pid for pid, _ in v]
            else:
                result[sid] = list(v)
        return result
    
    # Track which S1 claims each pool_id and with what score
    pool_claims = defaultdict(list)  # pool_id -> [(s1_id, score)]
    
    for sid, v in pred_dict.items():
        if not v:
            continue
        if isinstance(v[0], tuple):
            for pid, score in v:
                pool_claims[pid].append((sid, score))
        else:
            for pid in v:
                pool_claims[pid].append((sid, 1.0))  # no score available, equal weight
    
    # For pool_ids claimed by multiple S1, keep only strongest
    excluded = defaultdict(set)  # s1_id -> set of pool_ids to remove
    conflicts = 0
    for pid, claims in pool_claims.items():
        if len(claims) > 1:
            conflicts += 1
            # sort by score desc, keep only the top claim
            claims_sorted = sorted(claims, key=lambda x: x[1], reverse=True)
            for sid, _ in claims_sorted[1:]:
                excluded[sid].add(pid)
    
    if conflicts:
        print(f"  Cross-source: {conflicts} pool IDs claimed by multiple S1, resolved.")
    
    # Build clean result
    result = {}
    for sid, v in pred_dict.items():
        if isinstance(v[0], tuple) if v else False:
            pids = [pid for pid, _ in v if pid not in excluded[sid]]
        else:
            pids = [pid for pid in v if pid not in excluded[sid]] if v else []
        result[sid] = pids
    return result
