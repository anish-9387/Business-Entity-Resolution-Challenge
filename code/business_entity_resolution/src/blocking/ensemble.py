import pandas as pd
from collections import defaultdict
from src.blocking.token_blocking import token_block
from src.blocking.phonetic_blocking import phonetic_block
from src.blocking.address_blocking import address_block
from src.blocking.embedding_blocking import embedding_block

def ensemble_block(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, max_candidates: int = 50):
    """
    Union of all 4 blocking channels. NO per-country hard filtering.
    Country is handled as a feature, not a blocking filter.
    
    Returns:
        candidates: dict[s1_id -> list[pool_id]]
        channel_evidence: dict[(s1_id, pool_id) -> set[channel_name]]
    """
    pool = pd.concat([s2_df, s3_df], ignore_index=True)
    
    print("  Running token blocking...")
    token_cands = token_block(s1_df, pool, max_candidates)
    print(f"    Token: avg {sum(len(v) for v in token_cands.values()) / max(len(token_cands), 1):.1f} candidates/S1")
    
    print("  Running phonetic blocking...")
    phonetic_cands = phonetic_block(s1_df, pool, max_candidates=30)
    print(f"    Phonetic: avg {sum(len(v) for v in phonetic_cands.values()) / max(len(phonetic_cands), 1):.1f} candidates/S1")
    
    print("  Running address blocking...")
    addr_cands = address_block(s1_df, pool, max_candidates=30)
    print(f"    Address: avg {sum(len(v) for v in addr_cands.values()) / max(len(addr_cands), 1):.1f} candidates/S1")
    
    print("  Running embedding blocking...")
    emb_cands = embedding_block(s1_df, pool, max_candidates=30)
    print(f"    Embedding: avg {sum(len(v) for v in emb_cands.values()) / max(len(emb_cands), 1):.1f} candidates/S1")
    
    # Union all channels + track evidence
    candidates = {}
    channel_evidence = {}
    
    for sid in s1_df["entity_id"]:
        all_cands = set()
        evidence = defaultdict(set)
        
        for name, cdict in [("token", token_cands), ("phonetic", phonetic_cands),
                            ("address", addr_cands), ("embedding", emb_cands)]:
            for pid in cdict.get(sid, set()):
                all_cands.add(pid)
                evidence[pid].add(name)
        
        # Rank by evidence count (more channels = higher priority), truncate
        cand_list = sorted(all_cands, key=lambda p: len(evidence[p]), reverse=True)
        if len(cand_list) > max_candidates:
            cand_list = cand_list[:max_candidates]
        
        candidates[sid] = cand_list
        for pid in cand_list:
            channel_evidence[(sid, pid)] = evidence[pid]
    
    total_cands = sum(len(v) for v in candidates.values())
    print(f"  Ensemble union: {total_cands} total candidates, avg {total_cands / max(len(candidates), 1):.1f}/S1")
    
    return candidates, channel_evidence
