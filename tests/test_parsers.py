"""Deterministic unit tests for IS-Academia Showroom, Registrations, and CV parsers."""

from __future__ import annotations

import pytest

from give_me_job_please.core.parsers.cv_parser import parse_cv_bytes
from give_me_job_please.core.parsers.isacademia import (
    detect_isacademia_file_type,
    parse_registrations_html,
    parse_showroom_html,
)
from give_me_job_please.errors import ParseError

SAMPLE_SHOWROOM_HTML = """<!DOCTYPE html>
<html>
<head><meta http-equiv="content-type" content="text/html; charset=windows-1252"></head>
<body>
<table class="grid">
<tbody>
  <tr>
    <td id="stage_308197177_5608120730">
      <a href='javascript:void(prtl.synchronize("ww_i_stageview=5608120730"));'>View</a>
    </td>
    <td infobulle="&lt;b&gt;MAN&#140;UVRE AI&#153;&lt;/b&gt;&lt;br/&gt;&lt;u&gt;Entreprise:&lt;/u&gt;&lt;p&gt;Zefire&lt;br/&gt;&lt;i&gt; France - 13000 Marseille&lt;/i&gt;&lt;/p&gt;&lt;u&gt;Description:&lt;/u&gt;&lt;p&gt;Build RAG pipelines &amp; agents.&lt;/p&gt;">MAN&#140;UVRE AI&#153;</td>
    <td>Zefire</td>
    <td>Paris</td>
    <td infobulle=" Climate change (SDG 13)&lt;br/&gt; Energy (SDG 7)"><img alt=" Climate change (SDG 13)&lt;br/&gt; Energy (SDG 7)"></td>
    <td>60677</td>
    <td>master project or Internship</td>
    <td>3</td>
    <td>2</td>
    <td>To find (if master project)</td>
    <td>24.11.2026</td>
  </tr>
  <tr>
    <td id="stage_308197177_5580175759">View</td>
    <td infobulle="&lt;b&gt;Applied AI UEFA&lt;/b&gt;&lt;br/&gt;&lt;u&gt;Responsable:&lt;/u&gt;&lt;p&gt;Svensson Ola&lt;/p&gt;">Applied AI UEFA</td>
    <td>UEFA</td>
    <td>Nyon, Switzerland</td>
    <td>&nbsp;</td>
    <td>60541</td>
    <td>PDM coordonné</td>
    <td>15</td>
    <td>1</td>
    <td>Prof. Svensson Ola</td>
    <td>11.11.2026</td>
  </tr>
</tbody>
</table>
</body>
</html>
""".encode("windows-1252")

SAMPLE_REGISTRATIONS_HTML = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<div id="B308198588_305" ewnet:portal="xslpath:EPFL.Stages.2016.listeInscrStagesWithAction:en;">
  <div id="BB308198588_305">
    <table class="grid">
      <tbody>
        <tr>
          <td><a href='javascript:void(prtl.synchronize("ww_i_stageview=5608120730"));'>View</a></td>
          <td>MANOEUVRE AI</td>
          <td>Zefire</td>
          <td>60677</td>
          <td>Pending</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
<div id="B999"></div>
</body>
</html>
""".encode("utf-8")


def test_showroom_detection_and_parsing() -> None:
    assert detect_isacademia_file_type(SAMPLE_SHOWROOM_HTML) == "showroom"
    offers = parse_showroom_html(SAMPLE_SHOWROOM_HTML)
    assert len(offers) == 2

    first = offers[0]
    assert first.uid == "isacademia:60677"
    assert first.stage_id == "5608120730"
    assert "Œ" in first.title and "™" in first.title
    assert first.company_hq == "France - 13000 Marseille"
    assert first.label_durable == ["Climate change (SDG 13)", "Energy (SDG 7)"]
    assert first.creation_date == "2026-11-24"
    assert first.has_description is True
    assert first.description == "Build RAG pipelines & agents."
    assert first.competition_ratio == 1.5

    # PDM coordonné without embedded description must parse cleanly
    pdm = offers[1]
    assert pdm.uid == "isacademia:60541"
    assert pdm.has_description is False
    assert pdm.description == ""
    assert pdm.prof == "Prof. Svensson Ola"


def test_registrations_detection_and_parsing() -> None:
    assert detect_isacademia_file_type(SAMPLE_REGISTRATIONS_HTML) == "registrations"
    regs = parse_registrations_html(SAMPLE_REGISTRATIONS_HTML)
    assert len(regs) == 1
    assert regs[0].internship_number == "60677"
    assert regs[0].stage_id == "5608120730"


def test_invalid_html_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        detect_isacademia_file_type(b"<html><body>Hello world</body></html>")


def test_cv_text_parser() -> None:
    cv = parse_cv_bytes(b"MSc Computer Science EPFL\nSkills: Python, Rust, Security, ML.", "cv.txt")
    assert "EPFL" in cv
    with pytest.raises(ParseError):
        parse_cv_bytes(b"short", "cv.txt")
