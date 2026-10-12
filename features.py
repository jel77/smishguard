"""Text cleaning and hand-made features for SmishGuard (shared by the app and the tests)."""
import re

URGENT = ["kailangan agad", "agad", "ngayon", "claim now", "urgent", "expire", "last chance", "within 24"]
MONEY = ["php", "₱", "peso", "prize", "premyo", "panalo", "nanalo", "won", "reward", "cash", "gcash", "maya", "bdo", "bpi", "account"]
HAND_NAMES = ["has_url", "short_url", "suspicious_tld", "urgency_count", "money_count", "length",
              "special_chars", "has_phone", "sender_is_number", "sender_is_alpha", "sender_unknown"]
# PH mobile numbers: 09xx xxx xxxx, +639..., 639...
PH_PHONE = re.compile(r"(?<!\d)(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")

# Links: http(s)://..., www...., or a bare domain ending in a known TLD (so "P500.00" or "Ref: SM2210." don't count)
TLDS = ("com|net|org|ph|gov|edu|info|biz|io|co|me|ly|cc|xyz|top|online|site|click|link|live|shop|store|"
        "app|icu|buzz|vip|tk|ml|ga|cf|gq|gd|gy")
URL_RE = re.compile(rf"https?://\S+|www\.\S+|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:{TLDS})\b(?:/\S*)?", re.I)
SHORTENERS = re.compile(r"\b(?:bit\.ly|tinyurl\.com|cutt\.ly|shorturl\.at|is\.gd|goo\.gl|rb\.gy|ow\.ly|tiny\.cc|s\.id|t\.me)\b", re.I)
# Cheap TLDs often used for throwaway scam domains (official PH sites use .com, .com.ph, .gov.ph)
SUSPICIOUS_TLD = re.compile(r"\.(?:xyz|top|cc|info|online|site|click|link|live|shop|icu|buzz|vip|tk|ml|ga|cf|gq)\b(?:/|$)", re.I)


def urls(t):
    return [m.group().rstrip(".,!?)") for m in URL_RE.finditer(t)]


def clean(t):
    # Replace each link with a URL token but keep its words ("gcash-verify.top" -> "URL gcash verify top"),
    # since words like "verify" inside scam domains are useful signals
    t = URL_RE.sub(lambda m: " URL " + re.sub(r"[^a-z0-9]+", " ", m.group().lower()) + " ", t.lower())
    return re.sub(r"\d+", " NUM ", t)


def sender_type(sender):
    """'number' (plain mobile number), 'alpha' (named ID like BDO or a short code like 2958) or 'unknown' (blank)."""
    s = str(sender or "").strip()
    if not s or s.lower() == "nan":
        return "unknown"
    if re.fullmatch(r"\+?[\d\s-]{7,}", s):
        return "number"
    return "alpha"


def hand(t, sender=""):
    l = t.lower()
    st_ = sender_type(sender)
    links = urls(t)
    return [
        int(bool(links)),
        int(any(SHORTENERS.search(u) for u in links)),
        int(any(SUSPICIOUS_TLD.search(u.split("?")[0]) for u in links)),
        sum(k in l for k in URGENT),
        sum(k in l for k in MONEY),
        len(t),
        sum(not c.isalnum() and not c.isspace() for c in t),
        int(bool(PH_PHONE.search(t))),
        int(st_ == "number"),
        int(st_ == "alpha"),
        int(st_ == "unknown"),
    ]
