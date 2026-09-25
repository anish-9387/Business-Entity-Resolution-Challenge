from src.features.name_features import compute_name_features
from src.features.address_features import compute_address_features
from src.features.semantic_features import compute_semantic_features
from src.features.metadata_features import compute_metadata_features
from src.features.blocking_features import compute_blocking_features

def compute_all_features(s1_row, s2_row, channel_evidence=None, s1_id=None, pool_id=None):
    """Compute full feature vector for a candidate pair."""
    f = {}
    f.update(compute_name_features(s1_row, s2_row))
    f.update(compute_address_features(s1_row, s2_row))
    f.update(compute_semantic_features(s1_row, s2_row))
    f.update(compute_metadata_features(s1_row, s2_row))
    if channel_evidence is not None and s1_id and pool_id:
        f.update(compute_blocking_features(s1_id, pool_id, channel_evidence))
    else:
        f.update({
            'blocking_channel_count': 0.0, 
            'block_by_token': 0.0, 
            'block_by_phonetic': 0.0, 
            'block_by_address': 0.0, 
            'block_by_embedding': 0.0
        })
    return f

def feature_pair(s1, s2):
    """Legacy compatibility wrapper."""
    return compute_all_features(s1, s2)
