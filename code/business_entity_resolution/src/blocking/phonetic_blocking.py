from collections import defaultdict, Counter
import pandas as pd
from src.normalization import normalize_text
from src.phonetic import phonetic_keys

def phonetic_block(s1_df: pd.DataFrame, pool_df: pd.DataFrame, max_candidates: int = 30) -> dict:
    """
    Build inverted index of phonetic keys for pool business names.
    For each S1, retrieve candidates sharing phonetic keys.
    Returns dict[s1_entity_id -> set[pool_entity_id]]
    """
    pool_ids = pool_df["entity_id"].tolist()
    pool_names = pool_df["business_name"].fillna("").tolist()
    
    # Build phonetic key -> set(pool_ids) index
    phon_inv = defaultdict(set)
    for pid, name in zip(pool_ids, pool_names):
        keys = phonetic_keys(name)
        for k in set(keys):
            if k:
                phon_inv[k].add(pid)
    
    # Filter huge posting lists
    phon_inv = {k: v for k, v in phon_inv.items() if len(v) <= 3000}
    
    result = {}
    for _, row in s1_df.iterrows():
        sid = row["entity_id"]
        name = row.get("business_name", "") or ""
        keys = phonetic_keys(name)
        if not keys:
            result[sid] = set()
            continue
        
        cand_counter = Counter()
        for k in set(keys):
            if k in phon_inv:
                for pid in phon_inv[k]:
                    cand_counter[pid] += 1
        
        top = set(pid for pid, _ in cand_counter.most_common(max_candidates))
        result[sid] = top
    
    return result
