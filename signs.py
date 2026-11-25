"""Warning signs and sender spoofing checks for one message, shown on the Analyze tab.

They explain a message in plain words. They don't change the models' probabilities or verdict."""
import re

from features import (MONEY, PH_PHONE, SHORTENERS, SUSPICIOUS_TLD, URGENT, keywords_found,
                      scrambled_words, sender_type, urls)

# Brands that scammers often pretend to be: the words that name them, and their official websites.
# A link counts as official when its domain is one of these or a subdomain of one (e.g. www.bdo.com.ph)
BRANDS = {
    "GCash": (["gcash"], ["gcash.com"]),
    "Maya": (["maya", "paymaya"], ["maya.ph", "paymaya.com"]),
    "BDO": (["bdo"], ["bdo.com.ph"]),
    "BPI": (["bpi"], ["bpi.com.ph"]),
    "Metrobank": (["metrobank"], ["metrobank.com.ph"]),
    "UnionBank": (["unionbank"], ["unionbankph.com"]),
    "Landbank": (["landbank"], ["landbank.com"]),
    "PNB": (["pnb"], ["pnb.com.ph"]),
    "Security Bank": (["securitybank", "security bank"], ["securitybank.com"]),
    "RCBC": (["rcbc"], ["rcbc.com"]),
    "China Bank": (["chinabank", "china bank"], ["chinabank.ph"]),
}
_BRAND_IN_TEXT = {b: re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, words)) + r")(?![a-z0-9])")
                  for b, (words, _) in BRANDS.items()}


def link_domain(link):
    """The domain of a link, lowercase, without "www." ("https://www.BDO.com.ph/x" -> "bdo.com.ph")."""
    host = re.sub(r"^[a-z]+://", "", link.lower()).split("/")[0].split("?")[0].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def is_official(domain, brand):
    return any(domain == d or domain.endswith("." + d) for d in BRANDS[brand][1])


def brand_of_sender(sender):
    """The brand a named sender claims to be ("BDO", "GCash-Alert" -> "GCash"), or None."""
    if sender_type(sender) != "alpha":
        return None
    s = re.sub(r"[^a-z0-9]", "", str(sender).lower())
    for brand, (words, _) in BRANDS.items():
        if any(w.replace(" ", "") in s for w in words):
            return brand
    return None


def brands_in_text(text):
    l = text.lower()
    return [b for b, pattern in _BRAND_IN_TEXT.items() if pattern.search(l)]


def _verb(items, one, many):
    return one if len(items) == 1 else many


def _quote(words):
    return ", ".join(f'"{w}"' for w in words)


def warning_signs(text, sender=""):
    """The warning signs found in one message, as a list of {"sign", "detail"} dicts."""
    signs = []
    links = urls(text)
    short = [u for u in links if SHORTENERS.search(u)]
    odd = [u for u in links if not SHORTENERS.search(u) and SUSPICIOUS_TLD.search(u.split("?")[0])]
    other = [u for u in links if u not in short and u not in odd]
    if short:
        signs.append({"sign": "Shortened link",
                      "detail": f"{_quote(short)} {_verb(short, 'hides', 'hide')} the real website behind the link."})
    if odd:
        signs.append({"sign": "Unusual web address",
                      "detail": f"{_quote(odd)} {_verb(odd, 'uses', 'use')} a domain ending that is common in throwaway "
                                "scam sites."})
    if other:
        signs.append({"sign": "Contains a link",
                      "detail": f"{_quote(other)}. Check that it is the company's official website before "
                                "opening it."})
    urgent = keywords_found(text, URGENT)
    if urgent:
        signs.append({"sign": "Urgent or threatening wording", "detail": f"Found {_quote(urgent)}."})
    money = keywords_found(text, MONEY)
    if money:
        signs.append({"sign": "Money, bank or prize words", "detail": f"Found {_quote(money)}."})
    if PH_PHONE.search(text):
        signs.append({"sign": "Mobile number in the message",
                      "detail": "The message gives a mobile number to call or text."})
    scrambled = scrambled_words(text)
    if scrambled:
        signs.append({"sign": "Disguised spelling",
                      "detail": f"{_quote(scrambled)} {_verb(scrambled, 'uses', 'use')} look-alike characters, which "
                                "scammers use to get past filters."})
    return signs


def spoofing_warning(text, sender=""):
    """A warning when the sender may be faked, as a {"title", "detail"} dict, or None.

    Two cases:
    1. A named sender such as GCash or BDO, with a link that isn't that brand's official website.
       Sender names can be faked, so the name alone doesn't prove where the message came from.
    2. A brand is named in the message, the sender is a plain mobile number, and the message has a
       link or urgent wording (asking someone to act, not just mentioning the brand)."""
    links = urls(text)
    brand = brand_of_sender(sender)
    if brand:
        unofficial = [link_domain(u) for u in links if not is_official(link_domain(u), brand)]
        if unofficial:
            official = " or ".join(BRANDS[brand][1])
            return {"title": "Possible sender spoofing",
                    "detail": f"The sender name \"{sender}\" can be faked by scammers. This message links to "
                              f"{_quote(dict.fromkeys(unofficial))}, which is not an official {brand} website "
                              f"({official})."}
    if sender_type(sender) == "number" and (links or keywords_found(text, URGENT)):
        named = brands_in_text(text)
        if named:
            return {"title": "Brand message from a personal number",
                    "detail": f"The message mentions {', '.join(named)} but comes from a personal mobile number "
                              f"({sender}). Banks and e-wallets usually send alerts from a named sender, not "
                              "a mobile number."}
    return None
