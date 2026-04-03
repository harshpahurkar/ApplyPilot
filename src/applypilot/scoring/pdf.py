"""Resume PDF generation: LaTeX compilation + legacy text-to-PDF fallback.

Primary path: compile .tex files (Jake's Resume template) to PDF via pdflatex.
Fallback path: parse structured text, render via HTML/CSS, export via Playwright.
"""

import logging
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from applypilot.config import TAILORED_DIR

log = logging.getLogger(__name__)


# ── LaTeX Compilation (primary path) ─────────────────────────────────────

def _normalize_unicode_for_latex(text: str) -> str:
    """Normalize unsupported Unicode to pdflatex-safe text."""
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
        "\u2190": "<-",
        "\u2192": "->",
        "\u2194": "<->",
        "\u21d4": "<=>",
        "\u21c4": "<->",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    normalized_chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch in {"\n", "\t"} or 32 <= code < 127:
            normalized_chars.append(ch)
            continue
        decomp = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
        normalized_chars.append(decomp if decomp else " ")

    normalized = "".join(normalized_chars)
    normalized = re.sub(r"[\t ]+", " ", normalized)
    return normalized


def _sanitize_latex(text: str) -> str:
    """Fix common LLM-generated LaTeX issues before compilation."""
    # Remove empty itemize environments (missing \item)
    text = re.sub(
        r"\\resumeItemListStart\s*\\resumeItemListEnd",
        "",
        text,
    )
    # Remove empty subheading lists
    text = re.sub(
        r"\\resumeSubHeadingListStart\s*\\resumeSubHeadingListEnd",
        "",
        text,
    )
    # Remove empty sections (section header followed immediately by another section or end)
    text = re.sub(
        r"\\section\{[^}]*\}\s*(?=\\section\{|\\end\{document\})",
        "",
        text,
    )
    return text


# ── Quality Gate: verify tex content before compilation ──────────────────

_PLACEHOLDER_PATTERNS = [
    re.compile(r"Targeted\s+.+?\s+Project\s+\d", re.IGNORECASE),
    re.compile(r"Platform\s+Project\s+\d", re.IGNORECASE),
    re.compile(r"Project\s+[12]}\s*\$\|\\$", re.IGNORECASE),
]

_GENERIC_BULLET_SIGS = [
    "reduced deployment errors by 50%",
    "improved api latency by 38%",
    "reduced regression cycle time by 35% by automating core tests",
]


def validate_tex_quality(tex_content: str) -> dict:
    """Pre-compilation quality gate that catches placeholder/generic content.

    Returns:
        {"passed": bool, "issues": list[str]}
    """
    issues: list[str] = []
    lower = tex_content.lower()

    # Check for placeholder project names
    for pat in _PLACEHOLDER_PATTERNS:
        match = pat.search(tex_content)
        if match:
            issues.append(f"Placeholder project name: '{match.group().strip()}'")

    # Check for generic fallback bullets (identical defaults)
    generic_count = sum(1 for sig in _GENERIC_BULLET_SIGS if sig in lower)
    if generic_count >= 2:
        issues.append(f"Found {generic_count} generic fallback bullets — resume may be untailored")

    # Check for tech stack dumping (>8 items in a single project tech line)
    for m in re.finditer(r"\\emph\{\\small\s+([^}]+)\}", tex_content):
        tech_list = [t.strip() for t in m.group(1).split(",") if t.strip()]
        if len(tech_list) > 8:
            issues.append(f"Tech stack dump ({len(tech_list)} items) in project: {', '.join(tech_list[:5])}...")

    # Check for duplicate bullets across projects/experience
    bullet_texts = re.findall(r"\\resumeItem\{([^}]{30,})\}", tex_content)
    seen: dict[str, int] = {}
    for bt in bullet_texts:
        key = bt.lower().strip()[:80]
        seen[key] = seen.get(key, 0) + 1
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        issues.append(f"Duplicate bullet(s) across sections: {len(dupes)} repeated")

    return {"passed": len(issues) == 0, "issues": issues}


def compile_latex_to_pdf(
    tex_path: Path, output_path: Path | None = None, clean: bool = True
) -> Path:
    """Compile a .tex file to PDF using pdflatex.

    Runs pdflatex twice (for proper cross-references) in a temp directory
    to avoid cluttering the source folder with .aux/.log files.

    Args:
        tex_path: Path to the .tex file to compile.
        output_path: Optional override for the output PDF path.
            Defaults to same name with .pdf extension.
        clean: Whether to remove auxiliary files after compilation.

    Returns:
        Path to the generated PDF file.

    Raises:
        FileNotFoundError: If pdflatex is not installed.
        RuntimeError: If compilation fails.
    """
    tex_path = Path(tex_path)
    if not tex_path.exists():
        raise FileNotFoundError(f"LaTeX source not found: {tex_path}")

    # Find pdflatex -- check PATH first, then known install locations
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        # Search common MiKTeX / TeX Live install locations
        import os
        _candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe",
            Path("C:/Program Files/MiKTeX/miktex/bin/x64/pdflatex.exe"),
            Path(os.path.expanduser("~")) / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "pdflatex.exe",
            Path("C:/texlive/2024/bin/windows/pdflatex.exe"),
            Path("C:/texlive/2025/bin/windows/pdflatex.exe"),
            Path("C:/texlive/2026/bin/windows/pdflatex.exe"),
        ]
        for cand in _candidates:
            if cand.exists():
                pdflatex = str(cand)
                log.info("Found pdflatex at fallback location: %s", pdflatex)
                break
    if not pdflatex:
        raise FileNotFoundError(
            "pdflatex not found. Install MiKTeX (https://miktex.org) or "
            "TeX Live (https://tug.org/texlive/) and ensure pdflatex is on PATH."
        )

    out = output_path or tex_path.with_suffix(".pdf")
    out = Path(out)

    # Compile in a temp directory to keep things clean
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tmp_tex = tmp / tex_path.name
        source_tex = tex_path.read_text(encoding="utf-8", errors="replace")
        sanitized = _sanitize_latex(_normalize_unicode_for_latex(source_tex))
        tmp_tex.write_text(sanitized, encoding="utf-8")

        cmd = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory", str(tmp),
            str(tmp_tex),
        ]

        # Run twice for cross-references
        for run in range(2):
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=str(tmp)
            )
            if result.returncode != 0 and run == 1:
                log_file = tmp / tex_path.with_suffix(".log").name
                log_content = ""
                if log_file.exists():
                    log_content = log_file.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"pdflatex failed (exit {result.returncode}).\n"
                    f"STDERR: {result.stderr[:500]}\n"
                    f"LOG (last 2000 chars): {log_content}"
                )

        # Copy PDF to final destination
        tmp_pdf = tmp / tex_path.with_suffix(".pdf").name
        if not tmp_pdf.exists():
            raise RuntimeError(f"pdflatex did not produce a PDF: {tmp_pdf}")

        shutil.copy2(str(tmp_pdf), str(out))

    log.info("PDF compiled: %s", out)
    return out


# ── Resume Parser ────────────────────────────────────────────────────────

def parse_resume(text: str) -> dict:
    """Parse a structured text resume into sections.

    Expects a format with header lines (name, title, location, contact)
    followed by ALL-CAPS section headers (SUMMARY, TECHNICAL SKILLS, etc.).

    Args:
        text: Full resume text.

    Returns:
        {"name": str, "title": str, "location": str, "contact": str, "sections": dict}
    """
    lines = [line.rstrip() for line in text.strip().split("\n")]

    # Header: first few lines before SUMMARY
    header_lines: list[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().upper() == "SUMMARY":
            body_start = i
            break
        if line.strip():
            header_lines.append(line.strip())

    name = header_lines[0] if len(header_lines) > 0 else ""
    title = header_lines[1] if len(header_lines) > 1 else ""
    # The header may have 3 or 4 lines depending on whether location is included
    location = ""
    contact = ""
    if len(header_lines) > 3:
        location = header_lines[2]
        contact = header_lines[3]
    elif len(header_lines) > 2:
        # Could be location or contact -- check for email/phone indicators
        if "@" in header_lines[2] or "|" in header_lines[2]:
            contact = header_lines[2]
        else:
            location = header_lines[2]

    # Split body into sections by ALL-CAPS headers
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in lines[body_start:]:
        stripped = line.strip()
        # Detect section headers (all caps, no leading dash/bullet, longer than 3 chars)
        if (
            stripped
            and stripped == stripped.upper()
            and not stripped.startswith("-")
            and len(stripped) > 3
            and not stripped.startswith("\u2022")
        ):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return {
        "name": name,
        "title": title,
        "location": location,
        "contact": contact,
        "sections": sections,
    }


def parse_skills(text: str) -> list[tuple[str, str]]:
    """Parse skills section into (category, value) pairs.

    Args:
        text: The TECHNICAL SKILLS section text.

    Returns:
        List of (category_name, skills_string) tuples.
    """
    skills: list[tuple[str, str]] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            cat, val = line.split(":", 1)
            skills.append((cat.strip(), val.strip()))
    return skills


def parse_entries(text: str) -> list[dict]:
    """Parse experience/project entries from section text.

    Args:
        text: The EXPERIENCE or PROJECTS section text.

    Returns:
        List of {"title": str, "subtitle": str, "bullets": list[str]} dicts.
    """
    entries: list[dict] = []
    lines = text.strip().split("\n")
    current: dict | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped.startswith("\u2022 "):
            if current:
                current["bullets"].append(stripped[2:].strip())
        elif current is None or (
            not stripped.startswith("-")
            and not stripped.startswith("\u2022")
            and len(current.get("bullets", [])) > 0
        ):
            # New entry
            if current:
                entries.append(current)
            current = {"title": stripped, "subtitle": "", "bullets": []}
        elif current and not current["subtitle"]:
            current["subtitle"] = stripped
        else:
            if current:
                current["bullets"].append(stripped)

    if current:
        entries.append(current)

    return entries


# ── HTML Template ────────────────────────────────────────────────────────

def build_html(resume: dict) -> str:
    """Build professional resume HTML from parsed data.

    Args:
        resume: Parsed resume dict from parse_resume().

    Returns:
        Complete HTML string ready for PDF rendering.
    """
    sections = resume["sections"]

    # Skills
    skills_html = ""
    if "TECHNICAL SKILLS" in sections:
        skills = parse_skills(sections["TECHNICAL SKILLS"])
        rows = ""
        for cat, val in skills:
            rows += f'<div class="skill-row"><span class="skill-cat">{cat}:</span> {val}</div>\n'
        skills_html = f'<div class="section"><div class="section-title">Technical Skills</div>{rows}</div>'

    # Experience
    exp_html = ""
    if "EXPERIENCE" in sections:
        entries = parse_entries(sections["EXPERIENCE"])
        items = ""
        for e in entries:
            bullets = "".join(f"<li>{b}</li>" for b in e["bullets"])
            subtitle = f'<div class="entry-subtitle">{e["subtitle"]}</div>' if e["subtitle"] else ""
            items += f'<div class="entry"><div class="entry-title">{e["title"]}</div>{subtitle}<ul>{bullets}</ul></div>'
        exp_html = f'<div class="section"><div class="section-title">Experience</div>{items}</div>'

    # Projects
    proj_html = ""
    if "PROJECTS" in sections:
        entries = parse_entries(sections["PROJECTS"])
        items = ""
        for e in entries:
            bullets = "".join(f"<li>{b}</li>" for b in e["bullets"])
            subtitle = f'<div class="entry-subtitle">{e["subtitle"]}</div>' if e["subtitle"] else ""
            items += f'<div class="entry"><div class="entry-title">{e["title"]}</div>{subtitle}<ul>{bullets}</ul></div>'
        proj_html = f'<div class="section"><div class="section-title">Projects</div>{items}</div>'

    # Education
    edu_html = ""
    if "EDUCATION" in sections:
        edu_text = sections["EDUCATION"].strip()
        edu_html = f'<div class="section"><div class="section-title">Education</div><div class="edu">{edu_text}</div></div>'

    # Summary
    summary_html = ""
    if "SUMMARY" in sections:
        summary_html = f'<div class="section"><div class="section-title">Summary</div><div class="summary">{sections["SUMMARY"].strip()}</div></div>'

    # Contact line parsing
    contact = resume["contact"]
    contact_parts = [p.strip() for p in contact.split("|")] if contact else []
    contact_html = " &nbsp;|&nbsp; ".join(contact_parts)

    # Location line (may be empty)
    location_html = f'<div class="location">{resume["location"]}</div>' if resume["location"] else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: 0.35in 0.5in;
}}
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}
body {{
    font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #1a1a1a;
}}
.header {{
    text-align: center;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 1.5px solid #2a7ab5;
}}
.name {{
    font-size: 18pt;
    font-weight: 700;
    color: #1a3a5c;
    letter-spacing: 0.5px;
}}
.title {{
    font-size: 10.5pt;
    color: #3a6b8c;
    margin: 1px 0;
}}
.location {{
    font-size: 9pt;
    color: #555;
}}
.contact {{
    font-size: 9pt;
    color: #444;
    margin-top: 1px;
}}
.contact a {{
    color: #2c3e50;
    text-decoration: none;
}}
.section {{
    margin-top: 5px;
}}
.section-title {{
    font-size: 10pt;
    font-weight: 700;
    color: #1a3a5c;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1.5px solid #2a7ab5;
    padding-bottom: 1px;
    margin-bottom: 3px;
}}
.summary {{
    font-size: 9.5pt;
    color: #333;
    line-height: 1.4;
}}
.skill-row {{
    font-size: 9.5pt;
    margin: 0;
    line-height: 1.35;
}}
.skill-cat {{
    font-weight: 600;
    color: #1a3a5c;
}}
.entry {{
    margin-bottom: 4px;
    break-inside: avoid;
}}
.entry-title {{
    font-weight: 600;
    font-size: 10pt;
    color: #1a3a5c;
}}
.entry-subtitle {{
    font-size: 9pt;
    color: #4a7a9b;
    font-style: italic;
    margin-bottom: 1px;
}}
ul {{
    margin-left: 14px;
    padding: 0;
}}
li {{
    font-size: 9.5pt;
    margin-bottom: 1px;
    line-height: 1.35;
}}
.edu {{
    font-size: 10pt;
}}
</style>
</head>
<body>
<div class="header">
    <div class="name">{resume['name']}</div>
    <div class="title">{resume['title']}</div>
    {location_html}
    <div class="contact">{contact_html}</div>
</div>
{summary_html}
{skills_html}
{exp_html}
{proj_html}
{edu_html}
</body>
</html>"""


# ── PDF Renderer ─────────────────────────────────────────────────────────

def render_pdf(html: str, output_path: str) -> None:
    """Render HTML to PDF using Playwright's headless Chromium.

    Args:
        html: Complete HTML string.
        output_path: Path to write the PDF file.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        page.pdf(
            path=output_path,
            format="Letter",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            print_background=True,
        )
        browser.close()


# ── Cover Letter PDF ─────────────────────────────────────────────────────

def compile_cover_letter_pdf(
    txt_path: Path, output_path: Path | None = None
) -> Path:
    """Compile a plain-text cover letter to PDF via pdflatex.

    Uses a minimal LaTeX document with clean, professional formatting.
    No resume parsing — just straight paragraph text.

    Args:
        txt_path: Path to the .txt cover letter file.
        output_path: Optional override for the output PDF path.

    Returns:
        Path to the generated PDF file.
    """
    txt_path = Path(txt_path)
    text = txt_path.read_text(encoding="utf-8").strip()

    # Escape LaTeX special characters
    def _esc(s: str) -> str:
        s = _normalize_unicode_for_latex(s)
        for ch in ("\\", "&", "%", "$", "#", "_", "{", "}"):
            s = s.replace(ch, f"\\{ch}")
        s = s.replace("~", r"\textasciitilde{}")
        s = s.replace("^", r"\textasciicircum{}")
        return s

    escaped = _esc(text)
    # Convert double newlines to paragraph breaks
    paragraphs = [p.strip() for p in escaped.split("\n\n") if p.strip()]
    body = "\n\n".join(paragraphs)

    tex_source = (
        r"\documentclass[11pt,letterpaper]{article}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        r"\usepackage{parskip}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\begin{document}" "\n\n"
        f"{body}\n\n"
        r"\end{document}" "\n"
    )

    # Write a temp .tex and compile
    tex_tmp = txt_path.with_suffix(".tex")
    tex_tmp.write_text(tex_source, encoding="utf-8")

    try:
        return compile_latex_to_pdf(tex_tmp, output_path or txt_path.with_suffix(".pdf"))
    finally:
        # Clean up the temp .tex (keep only the .txt and .pdf)
        if tex_tmp.exists():
            tex_tmp.unlink()


# ── Public API ───────────────────────────────────────────────────────────

def convert_to_pdf(
    source_path: Path, output_path: Path | None = None, html_only: bool = False
) -> Path:
    """Convert a resume file (.tex or .txt) to PDF.

    For .tex files: compiles via pdflatex (preferred path).
    For .txt files: parses text, renders via HTML, exports via Playwright (legacy).

    Args:
        source_path: Path to the .tex or .txt file to convert.
        output_path: Optional override for the output path.
        html_only: If True and source is .txt, output HTML instead of PDF.

    Returns:
        Path to the generated PDF (or HTML) file.
    """
    source_path = Path(source_path)

    # LaTeX path (primary)
    if source_path.suffix == ".tex":
        try:
            return compile_latex_to_pdf(source_path, output_path)
        except Exception as exc:
            txt_fallback = source_path.with_suffix(".txt")
            if txt_fallback.exists():
                log.warning(
                    "LaTeX compile failed for %s, falling back to text render: %s",
                    source_path.name,
                    exc,
                )
                fallback_out = output_path or source_path.with_suffix(".pdf")
                return convert_to_pdf(txt_fallback, fallback_out, html_only=html_only)
            raise

    # Legacy text path (fallback)
    text = source_path.read_text(encoding="utf-8")
    resume = parse_resume(text)
    html = build_html(resume)

    if html_only:
        out = output_path or source_path.with_suffix(".html")
        out = Path(out)
        out.write_text(html, encoding="utf-8")
        log.info("HTML generated: %s", out)
        return out

    out = output_path or source_path.with_suffix(".pdf")
    out = Path(out)
    render_pdf(html, str(out))
    log.info("PDF generated: %s", out)
    return out


def batch_convert(limit: int = 50) -> int:
    """Convert .tex and .txt files in TAILORED_DIR that don't have corresponding PDFs.

    Prefers .tex files (LaTeX compilation). Falls back to .txt (HTML rendering).

    Args:
        limit: Maximum number of files to convert.

    Returns:
        Number of PDFs generated.
    """
    if not TAILORED_DIR.exists():
        log.warning("Tailored directory does not exist: %s", TAILORED_DIR)
        return 0

    # Collect .tex files first (preferred), then .txt as fallback
    tex_files = sorted(TAILORED_DIR.glob("*.tex"))
    txt_files = sorted(TAILORED_DIR.glob("*.txt"))

    # Exclude _JOB.txt files
    candidates: list[Path] = list(tex_files) + [
        f for f in txt_files
        if not f.name.endswith("_JOB.txt")
        # Skip .txt if a .tex with the same stem exists
        and not (TAILORED_DIR / f"{f.stem}.tex").exists()
    ]

    # Filter to those without a corresponding PDF
    to_convert: list[Path] = []
    for f in candidates:
        pdf_path = f.with_suffix(".pdf")
        if not pdf_path.exists():
            to_convert.append(f)
        if len(to_convert) >= limit:
            break

    if not to_convert:
        log.info("All files already have PDFs.")
        return 0

    log.info("Converting %d files to PDF...", len(to_convert))
    converted = 0
    for f in to_convert:
        try:
            convert_to_pdf(f)
            converted += 1
        except Exception as e:
            log.error("Failed to convert %s: %s", f.name, e)

    log.info("Done: %d/%d PDFs generated in %s", converted, len(to_convert), TAILORED_DIR)
    return converted
