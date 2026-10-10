"""Every figure in reports/number_ledger.md must appear, verbatim, in the section it cites."""

import re

import pytest

from exptools.data import REPO_ROOT

REPORTS = REPO_ROOT / "reports"
PIPE = "\x00"  # stands in for escaped pipes while a table row is split


def _ledger_rows() -> list[dict]:
    rows = []
    for line in (REPORTS / "number_ledger.md").read_text().splitlines():
        if not re.match(r"^\| H\d", line):
            continue
        cells = [c.strip() for c in line.replace(r"\|", PIPE).strip("|").split("|")]
        ident, _, value, source, section = cells
        rows.append(
            {"id": ident, "value": value.strip("`").replace(PIPE, "|"), "source": source, "section": section}
        )
    return rows


def _section_text(source: str, section: str) -> str:
    """Text from the heading containing `section` up to the next heading of the same or higher level."""
    lines = (REPORTS / source).read_text().splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"^(#+) ", line)
        if match and section in line:
            level = len(match.group(1))
            end = next(
                (j for j in range(i + 1, len(lines)) if re.match(rf"^#{{1,{level}}} ", lines[j])), len(lines)
            )
            return "\n".join(lines[i:end])
    raise AssertionError(f"no heading containing {section!r} in {source}")


ROWS = _ledger_rows()


def test_the_ledger_is_not_empty_and_ids_are_unique():
    assert len(ROWS) >= 50
    assert len({r["id"] for r in ROWS}) == len(ROWS)


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_value_appears_in_its_cited_section(row):
    assert row["value"] in _section_text(row["source"], row["section"])
