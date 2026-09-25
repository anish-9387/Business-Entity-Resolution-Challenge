try:
    from metaphone import doublemetaphone
    HAS_METAPHONE = True
except ImportError:
    HAS_METAPHONE = False

def _simple_soundex(word: str) -> str:
    """Basic Soundex implementation as fallback."""
    if not word:
        return ""
    word = word.upper()
    soundex = word[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    prev = mapping.get(word[0], '0')
    for ch in word[1:]:
        code = mapping.get(ch, '0')
        if code != '0' and code != prev:
            soundex += code
            if len(soundex) == 4:
                break
        prev = code
    return soundex.ljust(4, '0')[:4]

def phonetic_keys(text: str) -> list:
    """Return list of phonetic codes for each token in normalized text."""
    from src.normalization import tokens as get_tokens
    toks = get_tokens(text)
    keys = []
    for t in toks:
        if not t or len(t) < 2:
            continue
        if HAS_METAPHONE:
            primary, secondary = doublemetaphone(t)
            if primary:
                keys.append(primary)
            if secondary and secondary != primary:
                keys.append(secondary)
        else:
            keys.append(_simple_soundex(t))
    return keys
