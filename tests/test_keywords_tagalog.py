"""Tests for the Filipino/Taglish keywords added to URGENT and MONEY in features.py."""
import pytest

from features import MONEY, URGENT, hand

URGENCY, MONEY_COUNT = 3, 4  # positions in the list returned by hand()

NEW_URGENT = ["i-verify", "i-update", "i-confirm", "i-claim", "mag-login", "na-lock", "naka-hold",
              "mawawala", "click here", "kindly", "final notice"]
NEW_MONEY = ["bonus", "deposit", "libre", "jackpot", "slot", "casino", "tumaya", "sabong",
             "rebate", "winner", "kumita", "puhunan", "giveaway", "ayuda"]


def test_lists_are_lowercase_and_unique():
    # hand() lowercases the message before matching, so uppercase keywords would never match
    for words in (URGENT, MONEY):
        assert all(w == w.lower() and w.strip() == w and w for w in words)
        assert len(words) == len(set(words))


@pytest.mark.parametrize("word", NEW_URGENT)
def test_new_urgent_words_are_counted(word):
    assert word in URGENT
    assert hand(f"Test message {word.upper()} test")[URGENCY] >= 1


@pytest.mark.parametrize("word", NEW_MONEY)
def test_new_money_words_are_counted(word):
    assert word in MONEY
    assert hand(f"Test message {word.upper()} test")[MONEY_COUNT] >= 1


def test_taglish_smishing_scores_higher():
    scam = "Kailangan agad i-verify ang GCash account mo, mawawala ang balance: gcash-verify.top"
    assert hand(scam)[URGENCY] >= 3
    gambling = "Libreng bonus! Mag-deposit at maglaro ng slot sa casino, may jackpot pa"
    assert hand(gambling)[MONEY_COUNT] >= 5


@pytest.mark.parametrize("text", [
    "Coffee tayo bukas after class? Libre ko na, may promo ako sa app.",
    "Pre, san tayo kakain mamaya? Nakasweldo ako.",
    "Ate, pa-send naman ng sched ng exam natin bukas. Salamat!",
])
def test_everyday_messages_have_no_urgency(text):
    assert hand(text)[URGENCY] == 0


def test_legit_bank_notification_has_no_urgency():
    msg = "PNB: Your PNB account ending 3380 was credited P5,500.00 on Oct 10."
    assert hand(msg)[URGENCY] == 0
