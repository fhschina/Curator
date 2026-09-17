import pytest

from eval.dedup.pair_construction.outcomes import canonicalize_url_v0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Example.COM:443/path?q=1#fragment", ("example.com", "https://example.com/path?q=1", True)),
        ("http://example.com:80/", ("example.com", "http://example.com/", True)),
        (None, (None, None, True)),
        ("not-a-url", (None, None, False)),
    ],
)
def test_conservative_url_canonicalization(raw: str | None, expected: tuple[str | None, str | None, bool]) -> None:
    assert canonicalize_url_v0(raw) == expected
