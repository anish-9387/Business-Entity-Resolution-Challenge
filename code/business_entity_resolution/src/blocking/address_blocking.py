from collections import defaultdict, Counter
import math
import pandas as pd
from src.normalization import normalize_address, extract_postal, extract_house_number, tokens

def address_block(s1_df: pd.DataFrame, pool_df: pd.DataFrame, max_candidates: int = 30) -> dict:
    """
    Multiple address blocking strategies unioned:
    - Postal code match
    - House number + shared street token
    - Rare address tokens (IDF-weighted)
    Returns dict[s1_entity_id -> set[pool_entity_id]]
    """
    pool_ids = pool_df["entity_id"].tolist()
    pool_addrs = pool_df["business_address"].fillna("").tolist()
    
    # Build indices
    postal_inv = defaultdict(set)
    house_street_inv = defaultdict(set)  # (house_num, street_token) -> pool_ids
    addr_token_inv = defaultdict(set)
    
    for pid, addr in zip(pool_ids, pool_addrs):
        norm = normalize_address(addr)
        toks = norm.split()
        
        pc = extract_postal(addr)
        if pc:
            postal_inv[pc].add(pid)
        
        hn = extract_house_number(addr)
        if hn and toks:
            for t in toks[:5]:  # first 5 tokens likely street-related
                if t != hn and len(t) >= 3:
                    house_street_inv[(hn, t)].add(pid)
        
        for t in set(toks):
            if t and len(t) >= 3:
                addr_token_inv[t].add(pid)
    
    # Filter huge posting lists for address tokens
    addr_token_inv = {k: v for k, v in addr_token_inv.items() if len(v) <= 2000}
    
    # IDF for address tokens
    N = max(len(pool_ids), 1)
    addr_idf = {t: math.log(N / (len(pids) + 1)) for t, pids in addr_token_inv.items()}
    
    result = {}
    for _, row in s1_df.iterrows():
        sid = row["entity_id"]
        addr = row.get("business_address", "") or ""
        norm = normalize_address(addr)
        toks = norm.split()
        
        cand_scores = Counter()
        
        # Postal code match (strong signal)
        pc = extract_postal(addr)
        if pc and pc in postal_inv:
            for pid in postal_inv[pc]:
                cand_scores[pid] += 3.0
        
        # House number + street token
        hn = extract_house_number(addr)
        if hn:
            for t in toks[:5]:
                key = (hn, t)
                if key in house_street_inv:
                    for pid in house_street_inv[key]:
                        cand_scores[pid] += 2.0
        
        # Rare address tokens
        tok_idf = [(t, addr_idf.get(t, 0)) for t in toks if t in addr_token_inv and len(t) >= 3]
        tok_idf.sort(key=lambda x: x[1], reverse=True)
        for t, w in tok_idf[:4]:
            for pid in addr_token_inv[t]:
                cand_scores[pid] += w * 0.5
        
        if not cand_scores:
            result[sid] = set()
            continue
        
        top = set(pid for pid, _ in cand_scores.most_common(max_candidates))
        result[sid] = top
    
    return result
