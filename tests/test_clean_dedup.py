from joblens.clean import (
    CleanPosting,
    clean_postings,
    detect_language,
    frequent_lines,
    html_to_text,
    normalize_whitespace,
    strip_boilerplate,
)
from joblens.dedup import dedup, exact_hash, minhash, shingles, true_jaccard
from joblens.sources import RawPosting


def raw(i, text, company="Acme", source="arbeitnow"):
    return RawPosting(
        source=source,
        source_id=str(i),
        url=f"u{i}",
        fetched_at="2026-11-01T00:00:00+00:00",
        title="Developer (m/w/d)",
        company=company,
        raw_text=text,
    )


def test_html_to_text_keeps_structure():
    t = html_to_text(
        "<h2>Profil</h2><p>a &amp; b<br>c</p><ul><li>Python</li><li>SQL</li></ul>"
        "<script>evil()</script>"
    )
    t = normalize_whitespace(t)
    assert t == "Profil\na & b\nc\n\n- Python\n\n- SQL"
    assert "evil" not in t
    assert html_to_text("plain text") == "plain text"


def test_normalize_whitespace():
    s = "a  b\r\n\n\n\n • item\t\tx  "
    assert normalize_whitespace(s) == "a b\n\n- item x"


def test_boilerplate_is_conservative():
    text = "Aufgaben\nWir freuen uns auf Ihre Bewerbung!\nFooter line for every posting\nGehalt 50.000 €"
    freq = {"footer line for every posting", "gehalt 50.000 €"}
    out, n = strip_boilerplate(text, freq)
    assert out == "Aufgaben\nGehalt 50.000 €"  # salary line kept despite being frequent
    assert n == 2


def test_frequent_lines():
    texts = [f"shared footer text line\nunique {i}" for i in range(30)]
    assert frequent_lines(texts, min_count=20) == {"shared footer text line"}


def test_language_detection():
    assert (
        detect_language("Wir suchen eine Pflegefachkraft mit Erfahrung und Deutschkenntnissen")
        == "de"
    )
    assert detect_language("We are looking for a developer with experience in Python") == "en"


def test_clean_postings_drops_short():
    rows = [
        raw(1, "<p>" + "We build things and you will work with the team. " * 10 + "</p>"),
        raw(2, "short"),
    ]
    out, stats = clean_postings(rows)
    assert stats.input == 2 and stats.output == 1 and stats.empty_dropped == 1
    assert out[0].language == "en" and out[0].title == "Developer (m/w/d)"


def cp(i, text, company="Acme", lang="en", source="arbeitnow"):
    return CleanPosting(
        source=source,
        source_id=str(i),
        url="",
        fetched_at="t",
        title="T",
        company=company,
        text=text,
        language=lang,
    )


BASE = " ".join(f"word{i}" for i in range(200))


def test_dedup_exact_and_near():
    rows = [
        cp(1, BASE),
        cp(2, BASE.upper() + "  "),  # exact after normalisation
        cp(3, BASE.replace("word199", "different")),  # near duplicate
        cp(4, " ".join(f"other{i}" for i in range(200))),
    ]
    kept, st = dedup(rows)
    assert [r.source_id for r in kept] == ["1", "4"]
    assert st.exact_removed == 1 and st.near_removed == 1 and st.output == 2


def test_minhash_estimates_jaccard():
    a = BASE
    b = " ".join(f"word{i}" for i in range(100, 300))
    est = minhash(a).jaccard(minhash(b))
    assert abs(est - true_jaccard(a, b)) < 0.1
    assert exact_hash("A  b") == exact_hash("a b")
    assert shingles("a b", k=5) == {"a b"} and shingles("", k=5) == set()
    assert true_jaccard("", "") == 1.0


def test_escaped_html_is_converted_to_text():
    from joblens.clean import html_to_text

    raw = "&lt;div class=&quot;intro&quot;&gt;&lt;p&gt;We build tools &amp;amp; more.&lt;/p&gt;&lt;/div&gt;"
    out = html_to_text(raw)
    assert "<" not in out and "&lt;" not in out
    assert "We build tools & more." in out


def test_entities_in_plain_text_are_decoded():
    from joblens.clean import html_to_text

    assert html_to_text("Sales &amp; Marketing&nbsp;Team") == "Sales & Marketing\xa0Team"
