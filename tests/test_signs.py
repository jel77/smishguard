"""Tests for the warning signs and sender spoofing checks in signs.py."""
import pytest

from features import keywords_found, MONEY, URGENT
from signs import brand_of_sender, is_official, link_domain, spoofing_warning, warning_signs


def sign_names(text, sender=""):
    return [s["sign"] for s in warning_signs(text, sender)]


# ---------- keyword matching ----------
@pytest.mark.parametrize("text, absent", [
    ("Kain tayo mamaya", "maya"),    # "maya" inside "mamaya"
    ("Maglaro tayo bukas", "laro"),  # "laro" inside "maglaro"
])
def test_keywords_do_not_match_inside_other_words(text, absent):
    assert absent not in keywords_found(text, MONEY)


def test_keywords_match_at_the_start_of_a_word():
    assert "maya" in keywords_found("Send it to my Maya account", MONEY)
    assert "slot" in keywords_found("Free SLOTS bonus", MONEY)  # plural still counts
    assert "deposit" in keywords_found("Na-deposit na ba?", MONEY)
    assert "i-verify" in keywords_found("Pakiverify, I-VERIFY agad", URGENT)


# ---------- warning signs ----------
def test_scam_example_lists_its_signs():
    names = sign_names("GCash: Na-lock ang account mo. I-verify agad dito: bit.ly/gc-verify", "+639561234567")
    assert names == ["Shortened link", "Urgent or threatening wording", "Money, bank or prize words"]


def test_unusual_domain_and_scrambled_spelling():
    names = sign_names("Play Sl0ts now at win-big.xyz, call 09171234567")
    assert "Unusual web address" in names
    assert "Disguised spelling" in names
    assert "Mobile number in the message" in names
    detail = next(s["detail"] for s in warning_signs("Play Sl0ts now") if s["sign"] == "Disguised spelling")
    assert '"Sl0ts"' in detail


def test_casual_message_has_no_signs():
    assert warning_signs("pre nasend ko na yung pambayad, check mo na lang") == []


def test_signs_quote_the_words_found():
    signs = {s["sign"]: s["detail"] for s in warning_signs("URGENT: claim now ang premyo mo")}
    assert '"urgent"' in signs["Urgent or threatening wording"]
    assert '"premyo"' in signs["Money, bank or prize words"]


# ---------- sender spoofing ----------
@pytest.mark.parametrize("link, domain", [
    ("https://www.BDO.com.ph/login", "bdo.com.ph"),
    ("gcash-secure.top/login", "gcash-secure.top"),
    ("http://m.gcash.com:443/x?y=1", "m.gcash.com"),
])
def test_link_domain(link, domain):
    assert link_domain(link) == domain


def test_official_domains_include_subdomains_only():
    assert is_official("gcash.com", "GCash")
    assert is_official("m.gcash.com", "GCash")
    assert not is_official("gcash.com.verify-now.top", "GCash")
    assert not is_official("fakegcash.com", "GCash")


@pytest.mark.parametrize("sender, brand", [
    ("GCash", "GCash"), ("BDO", "BDO"), ("BDO-Alert", "BDO"), ("SecurityBank", "Security Bank"),
    ("09171234567", None), ("", None), ("Mama", None),
])
def test_brand_of_sender(sender, brand):
    assert brand_of_sender(sender) == brand


def test_named_sender_with_unofficial_link_is_flagged():
    warning = spoofing_warning("GCash: Your account is locked. Verify at gcash-secure.top/login", "GCash")
    assert warning["title"] == "Possible sender spoofing"
    assert "gcash-secure.top" in warning["detail"] and "gcash.com" in warning["detail"]


def test_named_sender_with_official_link_is_not_flagged():
    assert spoofing_warning("Get the new GCash app at https://www.gcash.com/download", "GCash") is None


def test_named_sender_without_a_link_is_not_flagged():
    assert spoofing_warning("Your BDO OTP is 482913. Do not share this code with anyone.", "BDO") is None


def test_brand_message_from_a_personal_number_is_flagged():
    warning = spoofing_warning("GCash: Na-lock ang account mo. I-verify agad dito: bit.ly/gc-verify",
                               "+639561234567")
    assert warning["title"] == "Brand message from a personal number"
    assert "GCash" in warning["detail"]


def test_friend_mentioning_a_brand_is_not_flagged():
    # A brand is named, but there is no link and no urgent wording
    assert spoofing_warning("Pre send ko na lang sa GCash mo mamaya", "09171234567") is None
