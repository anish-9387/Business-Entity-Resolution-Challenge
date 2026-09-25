import math

def compute_entity_features(scores_list: list) -> dict[str, float]:
    f = {}
    n = len(scores_list)
    f['entity_candidate_count'] = float(n)
    
    if n == 0:
        f['entity_top_score'] = 0.0
        f['entity_second_score'] = 0.0
        f['entity_score_margin'] = 0.0
        f['entity_score_gap_from_top'] = 0.0
        f['entity_mean_score'] = 0.0
        f['entity_std_score'] = 0.0
        f['entity_num_above_05'] = 0.0
        f['entity_num_above_08'] = 0.0
        return f
        
    scores = [score for _, score in scores_list]
    scores.sort(reverse=True)
    
    top = scores[0]
    second = scores[1] if n > 1 else 0.0
    third = scores[2] if n > 2 else 0.0
    
    f['entity_top_score'] = float(top)
    f['entity_second_score'] = float(second)
    f['entity_score_margin'] = float(top - second)
    f['entity_score_gap_from_top'] = float(top - third) if n >= 3 else 0.0
    
    mean = sum(scores) / n
    f['entity_mean_score'] = float(mean)
    
    if n > 1:
        variance = sum((s - mean) ** 2 for s in scores) / n
        f['entity_std_score'] = float(math.sqrt(variance))
    else:
        f['entity_std_score'] = 0.0
        
    f['entity_num_above_05'] = float(sum(1 for s in scores if s > 0.5))
    f['entity_num_above_08'] = float(sum(1 for s in scores if s > 0.8))
    
    return f
