"""Generate enriched UTF-8-BOM CSV and companion JSON exports."""

from __future__ import annotations

import csv
import io
import json

from give_me_job_please.core.models import EnrichedOffer

# Leading characters that make Excel / LibreOffice interpret a cell as a formula.
_FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r", "\n")


def _csv_safe(value: object) -> object:
    """Neutralise spreadsheet formula injection in string cells (OWASP CSV injection)."""
    if not isinstance(value, str) or not value:
        return value
    if value.lstrip(" ")[:1] in _FORMULA_TRIGGERS:
        return f"'{value}"
    return value


def export_csv_bytes(items: list[EnrichedOffer]) -> bytes:
    """Serialize enriched offers to UTF-8 with BOM CSV (Excel-safe accents)."""
    fieldnames = [
        "uid",
        "stage_id",
        "internship_number",
        "status",
        "is_starred",
        "score",
        "score_engine",
        "location_match",
        "internship",
        "company",
        "company_hq",
        "assignment_location",
        "label_durable",
        "format",
        "registered",
        "places",
        "competition_ratio",
        "prof",
        "creation_date",
        "has_description",
        "match_summary",
        "tags",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        o = item.offer
        row: dict[str, object] = {
            "uid": o.uid,
            "stage_id": o.stage_id,
            "internship_number": o.external_id,
            "status": item.status.value,
            "is_starred": item.is_starred,
            "score": item.score if item.score is not None else "",
            "score_engine": item.score_engine or "",
            "location_match": item.location_match if item.location_match is not None else "",
            "internship": o.title,
            "company": o.company,
            "company_hq": o.company_hq,
            "assignment_location": o.assignment_location,
            "label_durable": "; ".join(o.label_durable),
            "format": o.format,
            "registered": o.registered if o.registered is not None else "",
            "places": o.places if o.places is not None else "",
            "competition_ratio": o.competition_ratio if o.competition_ratio is not None else "",
            "prof": o.prof,
            "creation_date": o.creation_date,
            "has_description": o.has_description,
            "match_summary": item.match_summary,
            "tags": "; ".join(item.tags),
        }
        writer.writerow({key: _csv_safe(value) for key, value in row.items()})
    return buf.getvalue().encode("utf-8-sig")


def export_descriptions_json_bytes(items: list[EnrichedOffer]) -> bytes:
    """Serialize full descriptions and metadata keyed by internship_number."""
    payload = {}
    for item in items:
        o = item.offer
        key = o.external_id or o.uid
        payload[key] = {
            "uid": o.uid,
            "stage_id": o.stage_id,
            "internship_number": o.external_id,
            "status": item.status.value,
            "is_starred": item.is_starred,
            "score": item.score,
            "score_engine": item.score_engine,
            "location_match": item.location_match,
            "match_summary": item.match_summary,
            "tags": item.tags,
            "internship": o.title,
            "company": o.company,
            "company_hq": o.company_hq,
            "assignment_location": o.assignment_location,
            "label_durable": o.label_durable,
            "format": o.format,
            "prof": o.prof,
            "creation_date": o.creation_date,
            "description": o.description,
        }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
