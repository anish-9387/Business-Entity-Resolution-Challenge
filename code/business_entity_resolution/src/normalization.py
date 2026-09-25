import re
import unicodedata

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "llc", "llp", "plc", "co", "company",
    "pte", "sa", "sarl", "gmbh", "ag", "srl", "bv", "nv",
}

LEGAL_FAMILIES = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "ltd": "ltd", "limited": "ltd",
    "pvt": "pvt", "private": "pvt",
    "llc": "llc", "llp": "llp", "plc": "plc",
    "co": "co", "company": "co",
    "pte": "pte",
    "sa": "sa", "sarl": "sa",
    "gmbh": "gmbh", "ag": "ag",
    "srl": "srl", "bv": "bv", "nv": "nv",
}

ABBR = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "cir": "circle", "hwy": "highway", "pkwy": "parkway", "sq": "square",
}

_LANDMARK_PATTERNS = re.compile(
    r'\b(near|nr\.?|opp\.?|opposite|behind|beside|next\s+to|adjacent|adj\.?|'
    r'landmark|facing|in\s+front\s+of)\b', re.IGNORECASE
)

def normalize_text(s: str) -> str:
    """NFKD + transliterate (unidecode if available, else strip combining marks) + lower + strip non-alnum + collapse whitespace."""
    if s is None or (isinstance(s, float)):
        s = str(s) if s is not None else ""
    s = str(s)
    if s.lower() == "nan" or not s.strip():
        return ""
    s = unicodedata.normalize("NFKD", s)
    if HAS_UNIDECODE:
        s = unidecode(s)
    else:
        s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str) -> list:
    return normalize_text(s).split()

def name_without_suffix(norm: str) -> str:
    toks = norm.split()
    filtered = [t for t in toks if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else norm

def extract_suffix_family(norm: str) -> str:
    """Return the family key for the first legal suffix found, or empty string."""
    for t in norm.split():
        if t in LEGAL_FAMILIES:
            return LEGAL_FAMILIES[t]
    return ""

def token_sort(s: str) -> str:
    return " ".join(sorted(normalize_text(s).split()))

def normalize_address(s: str) -> str:
    n = normalize_text(s)
    toks = n.split()
    toks = [ABBR.get(t, t) for t in toks]
    return " ".join(toks)

def extract_postal(address: str) -> str:
    if not address or str(address).lower() == "nan":
        return ""
    m = re.search(r'\b\d{5,6}\b', str(address))
    return m.group(0) if m else ""

def extract_house_number(address: str) -> str:
    if not address or str(address).lower() == "nan":
        return ""
    n = normalize_text(address)
    m = re.match(r'^(\d+)', n)
    return m.group(1) if m else ""

def detect_landmark(address: str) -> bool:
    if not address or str(address).lower() == "nan":
        return False
    return bool(_LANDMARK_PATTERNS.search(str(address)))
