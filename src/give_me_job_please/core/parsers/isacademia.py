"""
Deterministic parser for EPFL IS-Academia `.htm` exports:
  1. Internships showroom (stu).htm  -> list[JobOffer]
  2. Registrations (stu).htm         -> list[RegistrationRecord]

Uses Python stdlib html.parser with auto-encoding detection (windows-1252 / utf-8),
numeric entity unescaping (&#140; -> Œ, &#153; -> ™), and hidden tooltip extraction.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Literal

from give_me_job_please.core.models import JobOffer, RegistrationRecord
from give_me_job_please.errors import ParseError

SHOWROOM_PARSER_VERSION = "isacademia_showroom:1.0"
REGISTRATIONS_PARSER_VERSION = "isacademia_registrations:1.0"


def detect_encoding(raw_bytes: bytes) -> str:
    """Detect HTML character encoding from <meta> charset, falling back to utf-8/windows-1252."""
    head = raw_bytes[:4096].decode("ascii", errors="ignore")
    match = re.search(r'charset\s*=\s*["\']?([a-zA-Z0-9._-]+)', head, re.IGNORECASE)
    if match:
        return match.group(1)
    try:
        raw_bytes.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "windows-1252"


def clean_text(value: str) -> str:
    """Unescape HTML entities, replace non-breaking spaces, and normalize whitespace."""
    text = html.unescape(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def strip_html_tags(fragment: str) -> str:
    """Strip HTML tags from tooltip fragments while preserving readable paragraph breaks."""
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"</p\s*>", "\n\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    text = html.unescape(fragment).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_iso_date(date_str: str) -> str:
    """Convert DD.MM.YYYY to YYYY-MM-DD for deterministic sorting."""
    date_str = date_str.strip()
    if not date_str:
        return ""
    try:
        return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
    except ValueError:
        return date_str


def parse_optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_infobulle(raw_infobulle: str) -> dict[str, str]:
    """Extract company HQ, full description, and academic supervisor from tooltip HTML."""
    if not raw_infobulle:
        return {"company_hq": "", "description": "", "responsable": ""}

    hq_match = re.search(r"<i\b[^>]*>(.*?)</i>", raw_infobulle, re.IGNORECASE | re.DOTALL)
    company_hq = clean_text(hq_match.group(1)) if hq_match else ""

    desc_match = re.search(
        r"<u\b[^>]*>\s*Description\s*:?\s*</u>\s*<p\b[^>]*>(.*?)</p>",
        raw_infobulle,
        re.IGNORECASE | re.DOTALL,
    )
    description = strip_html_tags(desc_match.group(1)) if desc_match else ""

    resp_match = re.search(
        r"<u\b[^>]*>\s*Responsable\s*:?\s*</u>\s*<p\b[^>]*>(.*?)</p>",
        raw_infobulle,
        re.IGNORECASE | re.DOTALL,
    )
    responsable = clean_text(resp_match.group(1)) if resp_match else ""

    return {
        "company_hq": company_hq,
        "description": description,
        "responsable": responsable,
    }


def parse_sdg_labels(raw_label: str) -> list[str]:
    """Split sustainability (`Label durable`) tooltip HTML into clean SDG tag list."""
    if not raw_label:
        return []
    parts = re.split(r"<br\s*/?>", raw_label, flags=re.IGNORECASE)
    return [clean_text(p) for p in parts if clean_text(p)]


def _extract_stage_id(td0: dict[str, Any], td1: dict[str, Any]) -> str:
    m = re.search(r"stage_\d+_(\d+)", td0.get("id", ""))
    if m:
        return m.group(1)
    m = re.search(r"ww_i_stage(?:view|inscr)=(\d+)", td1.get("onclick", "") or td0.get("hrefs", ""))
    return m.group(1) if m else ""


class _GridTableExtractor(HTMLParser):
    """Streaming parser extracting tbody rows from IS-Academia tablercc/grid tables."""

    def __init__(self, *, require_stage_td_id: bool) -> None:
        super().__init__(convert_charrefs=True)
        self._require_stage_td_id = require_stage_td_id
        self._in_target_table = False
        self._table_depth = 0
        self._in_tbody = False
        self._in_tr = False
        self._in_td = False
        self._current_row: list[dict[str, Any]] = []
        self._current_cell: dict[str, Any] | None = None
        self.rows: list[list[dict[str, Any]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag == "table":
            classes = attr_dict.get("class", "").split()
            if not self._in_target_table and (
                "grid" in classes or "ewnet:tablercc" in attr_dict
            ):
                self._in_target_table = True
                self._table_depth = 1
                return
            if self._in_target_table:
                self._table_depth += 1
                return

        if not self._in_target_table:
            return

        if tag == "tbody" and self._table_depth == 1:
            self._in_tbody = True
        elif tag == "tr" and self._in_tbody and self._table_depth == 1:
            self._in_tr = True
            self._current_row = []
        elif tag == "td" and self._in_tr and self._table_depth == 1:
            self._in_td = True
            self._current_cell = {
                "id": attr_dict.get("id", ""),
                "infobulle": attr_dict.get("infobulle", ""),
                "onclick": attr_dict.get("onclick", ""),
                "hrefs": "",
                "img_alt": "",
                "text_parts": [],
            }
        elif tag == "a" and self._in_td and self._current_cell is not None:
            href = attr_dict.get("href", "")
            onclick = attr_dict.get("onclick", "")
            self._current_cell["hrefs"] += f" {href} {onclick}"
        elif tag == "img" and self._in_td and self._current_cell is not None:
            if not self._current_cell["img_alt"]:
                self._current_cell["img_alt"] = (
                    attr_dict.get("infobulle") or attr_dict.get("alt") or ""
                )

    def handle_endtag(self, tag: str) -> None:
        if not self._in_target_table:
            return

        if tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0:
                self._in_target_table = False
                self._in_tbody = False
        elif tag == "tbody" and self._table_depth == 1:
            self._in_tbody = False
        elif tag == "tr" and self._in_tr and self._table_depth == 1:
            self._in_tr = False
            if self._require_stage_td_id:
                if len(self._current_row) >= 11 and self._current_row[0]["id"].startswith("stage_"):
                    self.rows.append(self._current_row)
            else:
                if len(self._current_row) >= 4:
                    self.rows.append(self._current_row)
            self._current_row = []
        elif tag == "td" and self._in_td and self._table_depth == 1:
            self._in_td = False
            if self._current_cell is not None:
                self._current_cell["text"] = clean_text("".join(self._current_cell["text_parts"]))
                del self._current_cell["text_parts"]
                self._current_row.append(self._current_cell)
                self._current_cell = None

    def handle_data(self, data: str) -> None:
        if self._in_td and self._current_cell is not None:
            self._current_cell["text_parts"].append(data)


def detect_isacademia_file_type(raw_bytes: bytes) -> Literal["showroom", "registrations"]:
    """Auto-detect whether an uploaded IS-Academia HTML snapshot is Showroom or Registrations."""
    encoding = detect_encoding(raw_bytes)
    text = raw_bytes.decode(encoding, errors="replace")

    # Strongest signal: showroom rows carry <td id="stage_<cellid>_<stageid>">
    if re.search(r'<td\s+id="stage_\d+_\d+"', text):
        return "showroom"

    # Registrations cell is typically visible (display: block) with listeInscrStagesWithAction
    if "listeInscrStagesWithAction" in text or "Current registrations" in text or "Inscriptions actuelles" in text:
        return "registrations"

    raise ParseError(
        "Unrecognized HTML file. Please save the EPFL IS-Academia "
        "'Internships showroom (stu)' or 'Registrations (stu)' page (Ctrl+S -> Webpage, Complete/HTML only).",
        detail="No IS-Academia showroom stage_ rows or registrations markers found.",
    )


def parse_showroom_html(raw_bytes: bytes) -> list[JobOffer]:
    """Deterministically parse an IS-Academia showroom .htm export into canonical JobOffers."""
    encoding = detect_encoding(raw_bytes)
    content = raw_bytes.decode(encoding, errors="replace")

    extractor = _GridTableExtractor(require_stage_td_id=True)
    extractor.feed(content)
    extractor.close()

    if not extractor.rows:
        raise ParseError(
            "No internship listings were found in this HTML file. Make sure the showroom table "
            "has loaded in IS-Academia before saving the page.",
            detail="GridTableExtractor returned 0 rows with id='stage_*'.",
        )

    offers: list[JobOffer] = []
    seen_uids: set[str] = set()

    for cells in extractor.rows:
        stage_id = _extract_stage_id(cells[0], cells[1])
        title = cells[1]["text"]
        company = cells[2]["text"]
        assignment_location = cells[3]["text"]
        sdg_raw = cells[4]["infobulle"] or cells[4]["img_alt"]
        sdg_labels = parse_sdg_labels(sdg_raw)
        internship_number = cells[5]["text"]
        fmt = cells[6]["text"]
        registered = parse_optional_int(cells[7]["text"])
        places = parse_optional_int(cells[8]["text"])
        prof = cells[9]["text"]
        creation_date = parse_iso_date(cells[10]["text"])

        tooltip = parse_infobulle(cells[1]["infobulle"])
        external_id = internship_number or stage_id
        if not external_id:
            continue

        uid = f"isacademia:{external_id}"
        if uid in seen_uids:
            continue
        seen_uids.add(uid)

        offers.append(
            JobOffer(
                uid=uid,
                source="isacademia",
                external_id=external_id,
                stage_id=stage_id,
                title=title,
                company=company,
                company_hq=tooltip["company_hq"],
                assignment_location=assignment_location,
                label_durable=sdg_labels,
                format=fmt,
                registered=registered,
                places=places,
                prof=prof or tooltip["responsable"],
                creation_date=creation_date,
                description=tooltip["description"],
                has_description=bool(tooltip["description"]),
                parser_version=SHOWROOM_PARSER_VERSION,
            )
        )

    return offers


def parse_registrations_html(raw_bytes: bytes) -> list[RegistrationRecord]:
    """
    Parse an IS-Academia Registrations (stu).htm export to extract offers already applied to.

    Isolates the registrations cell (`BB308198588_305` / `listeInscrStagesWithAction`) when
    present, then extracts any 5-digit internship numbers, stage_ids, and title/company pairs.
    """
    encoding = detect_encoding(raw_bytes)
    content = raw_bytes.decode(encoding, errors="replace")

    # Narrow to the registrations cell block if the full portal page was saved
    block_match = re.search(
        r'id="BB\d+_305"[^>]*>(.*?)</div>\s*</div>\s*<div\s+id="B',
        content,
        re.IGNORECASE | re.DOTALL,
    )
    target_html = block_match.group(1) if block_match else content

    extractor = _GridTableExtractor(require_stage_td_id=False)
    extractor.feed(target_html)
    extractor.close()

    records: list[RegistrationRecord] = []
    seen_keys: set[str] = set()

    for row in extractor.rows:
        texts = [c["text"] for c in row]
        href_blob = " ".join(c.get("hrefs", "") + " " + c.get("onclick", "") for c in row)

        stage_match = re.search(r"ww_i_stage(?:view|inscr)?=(\d+)", href_blob)
        stage_id = stage_match.group(1) if stage_match else ""

        # Internship numbers in EPFL IS-Academia are 5-6 digit integers (e.g. 60524)
        internship_number = ""
        for t in texts:
            if re.fullmatch(r"\d{5,6}", t):
                internship_number = t
                break

        if not internship_number and not stage_id:
            continue

        dedup_key = internship_number or stage_id
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        non_empty = [t for t in texts if t and t != internship_number]
        records.append(
            RegistrationRecord(
                internship_number=internship_number,
                stage_id=stage_id,
                title=non_empty[0] if len(non_empty) > 0 else "",
                company=non_empty[1] if len(non_empty) > 1 else "",
                status_text=non_empty[-1] if len(non_empty) > 2 else "",
            )
        )

    return records
