import math
from collections import Counter
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from src.normalization import (
    normalize_text,
    tokens,
    name_without_suffix,
    extract_suffix_family,
    token_sort
)
from src.phonetic import phonetic_keys

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

def lcs_length(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_name_features(s1_row, s2_row) -> dict[str, float]:
    n1_raw = str(s1_row.get('business_name', ''))
    n2_raw = str(s2_row.get('business_name', ''))
    
    n1 = normalize_text(n1_raw)
    n2 = normalize_text(n2_raw)
    
    n1_nosuf = name_without_suffix(n1)
    n2_nosuf = name_without_suffix(n2)
    
    tok1 = tokens(n1)
    tok2 = tokens(n2)
    tok1_nosuf = tokens(n1_nosuf)
    tok2_nosuf = tokens(n2_nosuf)
    
    f = {}
    f['name_exact'] = 1.0 if (n1 and n2 and n1 == n2) else 0.0
    f['name_exact_nosuf'] = 1.0 if (n1_nosuf and n2_nosuf and n1_nosuf == n2_nosuf) else 0.0
    
    f['name_levenshtein'] = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    f['name_levenshtein_nosuf'] = fuzz.ratio(n1_nosuf, n2_nosuf) / 100.0 if n1_nosuf and n2_nosuf else 0.0
    
    f['name_jaro_winkler'] = JaroWinkler.similarity(n1, n2) if n1 and n2 else 0.0
    
    f['name_token_set'] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    f['name_token_sort'] = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    f['name_token_set_nosuf'] = fuzz.token_set_ratio(n1_nosuf, n2_nosuf) / 100.0 if n1_nosuf and n2_nosuf else 0.0
    
    set1, set2 = set(tok1), set(tok2)
    set1_ns, set2_ns = set(tok1_nosuf), set(tok2_nosuf)
    
    f['name_jaccard'] = float(len(set1 & set2) / len(set1 | set2)) if (set1 | set2) else 0.0
    f['name_jaccard_nosuf'] = float(len(set1_ns & set2_ns) / len(set1_ns | set2_ns)) if (set1_ns | set2_ns) else 0.0
    
    shared = set1 & set2
    f['name_shared_tokens'] = float(len(shared))
    f['name_shared_rare_tokens'] = float(len([t for t in shared if len(t) >= 4]))
    
    len1, len2 = len(n1), len(n2)
    max_len = max(len1, len2, 1)
    f['name_len_diff'] = float(abs(len1 - len2) / max_len)
    f['name_len_ratio'] = float(min(len1, len2) / max_len)
    
    f['name_token_count_diff'] = float(abs(len(tok1) - len(tok2)))
    
    lcs = lcs_length(n1, n2)
    f['name_lcs_ratio'] = float(lcs / max_len)
    
    f['name_char3_cosine'] = get_char3_cosine(n1, n2)
    
    pk1, pk2 = set(phonetic_keys(n1_raw)), set(phonetic_keys(n2_raw))
    f['name_phonetic_agreement'] = float(len(pk1 & pk2) / len(pk1 | pk2)) if (pk1 | pk2) else 0.0
    
    fam1 = extract_suffix_family(n1)
    fam2 = extract_suffix_family(n2)
    
    f['legal_suffix_same_family'] = 1.0 if (fam1 and fam2 and fam1 == fam2) else 0.0
    f['legal_suffix_both_present'] = 1.0 if (fam1 and fam2) else 0.0
    f['legal_suffix_conflict'] = 1.0 if (fam1 and fam2 and fam1 != fam2) else 0.0
    
    return f
