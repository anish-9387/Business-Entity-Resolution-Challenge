def apply_singleton_filter(s1_id, candidates_with_scores, threshold, margin=0.0):
    """
    Apply singleton-aware decision for one S1 entity.
    
    candidates_with_scores: list of (pool_id, score) tuples, sorted by score desc
    threshold: minimum score to accept
    margin: minimum gap between top and ambiguous candidates
    
    Returns: list of accepted pool_ids
    """
    if not candidates_with_scores:
        return []  # True singleton (no candidates)
    
    sorted_cands = sorted(candidates_with_scores, key=lambda x: x[1], reverse=True)
    top_score = sorted_cands[0][1]
    
    # Singleton detection: if even the best candidate is weak, return empty
    if top_score < threshold:
        return []
    
    second_score = sorted_cands[1][1] if len(sorted_cands) > 1 else 0.0
    
    accepted = []
    for pid, score in sorted_cands:
        if score < threshold:
            break
        # Margin rule: if multiple candidates are very close to top, be conservative
        if margin > 0 and len(sorted_cands) > 1:
            if score < top_score and (top_score - score) < margin:
                # Ambiguous zone: require higher confidence
                if score >= threshold + 0.05:
                    accepted.append(pid)
            else:
                accepted.append(pid)
        else:
            accepted.append(pid)
    
    return accepted
