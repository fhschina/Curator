import pytest

from eval.dedup.core.contracts import cp1_pair, stable_record_id


def test_canonical_pair_is_order_independent_and_domain_separated() -> None:
    forward = cp1_pair(10, 2)
    reverse = cp1_pair(2, 10)

    assert forward == reverse
    assert forward.doc_id_low == "10"
    assert forward.doc_id_high == "2"
    assert forward.canonical_pair_id.startswith("cp1_")
    assert stable_record_id("first", 1) != stable_record_id("second", 1)


def test_canonical_pair_rejects_self_pair() -> None:
    with pytest.raises(ValueError, match="self-pairs"):
        cp1_pair("same", "same")
