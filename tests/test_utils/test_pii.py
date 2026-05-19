"""PII redaction tests."""

from __future__ import annotations

from utils.pii import redact, redact_dict


def test_email_masked() -> None:
    assert redact("contact me at alice@example.com please") == "contact me at [EMAIL] please"


def test_ssn_masked() -> None:
    assert "[SSN]" in redact("SSN is 123-45-6789")


def test_phone_masked() -> None:
    out = redact("call +1 415 555 1234 tomorrow")
    assert "[PHONE]" in out


def test_credit_card_luhn_validated() -> None:
    # Valid Luhn: 4111 1111 1111 1111 (test Visa)
    assert "[CC]" in redact("card 4111 1111 1111 1111")
    # Not a valid card — phone-like 7 digit sequence should not become [CC]
    out = redact("ticket 1234567")
    assert "[CC]" not in out


def test_ip_masked() -> None:
    assert "[IP]" in redact("server at 10.0.0.42")


def test_redact_dict_walks_nested() -> None:
    data = {
        "query": "ping alice@example.com",
        "nested": {"phone": "+1 415 555 1234", "ok": "no pii here"},
        "list": ["nothing", "bob@test.com"],
    }
    out = redact_dict(data)
    assert "[EMAIL]" in out["query"]
    assert "[PHONE]" in out["nested"]["phone"]
    assert "[EMAIL]" in out["list"][1]
    assert out["nested"]["ok"] == "no pii here"


def test_redact_handles_empty() -> None:
    assert redact("") == ""
    assert redact_dict({}) == {}
