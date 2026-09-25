import math
from collections import Counter
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from src.normalization import (
    normalize_address,
    tokens,
    extract_postal,
    extract_house_number,
    detect_landmark
)

def get_char_ngrams(text, n=3):
    return [text[i:i+n] for i in range(len(text)-n+1)] if len(text) >= n else [text] if text else []

def get_char3_cosine(s1, s2):
    if not s1 or not s2:
        return 0.0
    c1 = Counter(get_char_ngrams(s1, 3))
    c2 = Counter(get_char_ngrams(s2, 3))
    intersection = set(c1.keys()) & set(c2.keys())
    numerator = sum([c1[x] * c2[x] for x in intersection])
    sum1 = sum([c1[x]**2 for x in c1.keys()])
    sum2 = sum([c2[x]**2 for x in c2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator / denominator) if denominator else 0.0

def compute_address_features(s1_row, s2_row) -> dict[str, float]:
    a1_raw = str(s1_row.get('business_address', ''))
    a2_raw = str(s2_row.get('business_address', ''))
    
    a1 = normalize_address(a1_raw)
    a2 = normalize_address(a2_raw)
    
    tok1 = tokens(a1)
    tok2 = tokens(a2)
    
    f = {}
    f['addr_levenshtein'] = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    f['addr_jaro_winkler'] = JaroWinkler.similarity(a1, a2) if a1 and a2 else 0.0
    f['addr_token_set'] = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    f['addr_token_sort'] = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    
    set1, set2 = set(tok1), set(tok2)
    f['addr_jaccard'] = float(len(set1 & set2) / len(set1 | set2)) if (set1 | set2) else 0.0
    f['addr_shared_tokens'] = float(len(set1 & set2))
    f['addr_char3_cosine'] = get_char3_cosine(a1, a2)
    
    p1 = extract_postal(a1_raw) or extract_postal(a1)
    p2 = extract_postal(a2_raw) or extract_postal(a2)
    
    f['postal_match'] = 1.0 if p1 and p2 and p1 == p2 else 0.0
    f['postal_both_missing'] = 1.0 if not p1 and not p2 else 0.0
    f['postal_one_missing'] = 1.0 if (not p1 and p2) or (p1 and not p2) else 0.0
    f['postal_prefix3_match'] = 1.0 if p1 and p2 and p1[:3] == p2[:3] else 0.0
    
    hn1 = extract_house_number(a1_raw) or extract_house_number(a1)
    hn2 = extract_house_number(a2_raw) or extract_house_number(a2)
    
    f['house_number_match'] = 1.0 if hn1 and hn2 and hn1 == hn2 else 0.0
    f['house_number_both_missing'] = 1.0 if not hn1 and not hn2 else 0.0
    
    len1, len2 = len(a1), len(a2)
    max_len = max(len1, len2, 1)
    f['addr_len_ratio'] = float(min(len1, len2) / max_len)
    
    lm1 = detect_landmark(a1_raw) or detect_landmark(a1)
    lm2 = detect_landmark(a2_raw) or detect_landmark(a2)
    
    f['landmark_either'] = 1.0 if lm1 or lm2 else 0.0
    f['landmark_both'] = 1.0 if lm1 and lm2 else 0.0
    
    return f
