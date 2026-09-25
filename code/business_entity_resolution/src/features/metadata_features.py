from rapidfuzz import fuzz
from src.normalization import normalize_text, normalize_address, detect_landmark

def compute_metadata_features(s1_row, s2_row) -> dict[str, float]:
    f = {}
    
    # pool record is s2_row in this context, assuming s1 is reference/query, s2 is pool candidate
    pool_id = str(s2_row.get('entity_id', ''))
    f['source_is_s2'] = 1.0 if pool_id.startswith('S2-') else 0.0
    f['source_is_s3'] = 1.0 if pool_id.startswith('S3-') else 0.0
    
    c1 = str(s1_row.get('country', '')).strip().lower()
    c2 = str(s2_row.get('country', '')).strip().lower()
    
    f['country_equal'] = 1.0 if c1 and c2 and c1 == c2 else 0.0
    f['country_either_missing'] = 1.0 if not c1 or not c2 else 0.0
    f['country_both_present'] = 1.0 if c1 and c2 else 0.0
    
    n1 = normalize_text(str(s1_row.get('business_name', '')))
    n2 = normalize_text(str(s2_row.get('business_name', '')))
    a1 = normalize_address(str(s1_row.get('business_address', '')))
    a2 = normalize_address(str(s2_row.get('business_address', '')))
    
    name_sim = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    addr_sim = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    
    f['high_name_low_addr'] = 1.0 if name_sim > 0.85 and addr_sim < 0.4 else 0.0
    f['low_name_high_addr'] = 1.0 if name_sim < 0.5 and addr_sim > 0.7 else 0.0
    f['name_addr_gap'] = float(name_sim - addr_sim)
    
    lm1 = detect_landmark(str(s1_row.get('business_address', ''))) or detect_landmark(a1)
    lm2 = detect_landmark(str(s2_row.get('business_address', ''))) or detect_landmark(a2)
    landmark_either = 1.0 if lm1 or lm2 else 0.0
    
    f['name_x_landmark'] = float(name_sim * landmark_either)
    
    return f
