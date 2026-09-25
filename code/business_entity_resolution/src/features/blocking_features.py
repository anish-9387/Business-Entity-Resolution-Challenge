def compute_blocking_features(s1_id: str, pool_id: str, channel_evidence: dict) -> dict[str, float]:
    f = {}
    channels = channel_evidence.get((s1_id, pool_id), set())
    
    f['blocking_channel_count'] = float(len(channels))
    f['block_by_token'] = 1.0 if 'token' in channels else 0.0
    f['block_by_phonetic'] = 1.0 if 'phonetic' in channels else 0.0
    f['block_by_address'] = 1.0 if 'address' in channels else 0.0
    f['block_by_embedding'] = 1.0 if 'embedding' in channels else 0.0
    
    return f
