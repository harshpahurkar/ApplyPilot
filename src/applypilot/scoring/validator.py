"""Resume and cover letter validation: banned words, fabrication detection, structural checks.

All validation is profile-driven -- no hardcoded personal data. The validator receives
a profile dict (from applypilot.config.load_profile()) and validates against the user's
actual skills, companies, projects, and school.
"""

import re
import logging

log = logging.getLogger(__name__)


# ── Universal Constants (not personal data) ───────────────────────────────

BANNED_WORDS: list[str] = [
    "passionate", "dedicated", "committed to",
    "utilizing", "utilize", "harnessing",
    "spearheaded", "spearhead", "orchestrated", "championed", "pioneered",
    "scalable solutions", "cutting-edge", "state-of-the-art", "best-in-class",
    "proven track record", "track record of success", "demonstrated ability",
    "strong communicator", "team player", "fast learner", "self-starter", "go-getter",
    "synergy", "cross-functional collaboration", "holistic",
    "transformative", "innovative solutions", "paradigm", "ecosystem",
    "proactive", "detail-oriented", "highly motivated",
    "full lifecycle",
    "deep understanding", "extensive experience", "comprehensive knowledge",
    "thrives in", "excels at", "adept at", "well-versed in",
    "i am confident", "i believe", "i am excited",
    "plays a critical role", "instrumental in", "integral part of",
    "strong track record", "eager to", "eager",
    # Cover-letter-specific additions
    "this demonstrates", "this reflects", "i have experience with",
    "furthermore", "additionally", "moreover",
]

# Words that are warnings (soft bans) -- demoted from hard ban because they
# can be legitimate engineering terms even though LLMs overuse them.
SOFT_BANNED_WORDS: list[str] = ["robust", "seamless", "leveraged", "leverage"]

LLM_LEAK_PHRASES: list[str] = [
    "i am sorry", "i apologize", "i will try", "let me try",
    "i am at a loss", "i am truly sorry", "apologies for",
    "i keep fabricating", "i will have to admit", "one final attempt",
    "one last time", "if it fails again", "persistent errors",
    "i am having difficulty", "i made an error", "my mistake",
    "here is the corrected", "here is the revised", "here is the updated",
    "here is my", "below is the", "as requested",
    "note:", "disclaimer:", "important:",
    "i have rewritten", "i have removed", "i have fixed",
    "i have replaced", "i have updated", "i have corrected",
    "per your feedback", "based on your feedback", "as per the instructions",
    "the following resume", "the resume below",
    "the following cover letter", "the letter below",
]

# Known fabrication markers: only certifications that cannot be faked.
# Languages, frameworks, and tools are ALLOWED if they appear in the JD
# (the JD exemption in validate_json_fields handles this).
# The candidate builds projects to match their resume before interviews,
# so any learnable tool from the JD is fair game for ATS optimization.
FABRICATION_WATCHLIST: set[str] = {
    # Hard lies: certifications can't be stretched -- these are verifiable
    "certif", "certified", "pmp", "scrum master", "aws certified",
    "azure certified", "gcp certified", "cka", "ckad", "cissp",
}

REQUIRED_SECTIONS: set[str] = {"SUMMARY", "TECHNICAL SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION"}


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_skills_set(profile: dict) -> set[str]:
    """Build the set of allowed skills from the profile's skills_boundary."""
    boundary = profile.get("skills_boundary", {})
    allowed: set[str] = set()
    for category in boundary.values():
        if isinstance(category, list):
            allowed.update(s.lower().strip() for s in category)
        elif isinstance(category, set):
            allowed.update(s.lower().strip() for s in category)
    return allowed


def sanitize_text(text: str) -> str:
    """Auto-fix common LLM output issues instead of rejecting."""
    text = text.replace(" \u2014 ", ", ").replace("\u2014", ", ")   # em dash -> comma
    text = text.replace("\u2013", "-")    # en dash -> hyphen
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # smart double quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")   # smart single quotes
    return text.strip()


def _normalize_validation_mode(validation_mode: str) -> str:
    """Normalize user-provided validation mode to a safe value."""
    mode = (validation_mode or "normal").strip().lower()
    if mode not in {"strict", "normal", "lenient"}:
        return "normal"
    return mode


# ── JSON Field Validation ─────────────────────────────────────────────────

def validate_json_fields(
    data: dict,
    profile: dict,
    jd_text: str = "",
    validation_mode: str = "normal",
) -> dict:
    """Validate individual JSON fields from an LLM-generated tailored resume.

    Supports the Jake's Resume LaTeX schema:
    - title, skills, experience, projects, education (education is a dict)
    - experience entries have: company, title, location, dates, bullets, skills_used
    - project entries have: name, tech_stack, dates, bullets, skills_used

    Args:
        data: Parsed JSON from the LLM.
        profile: User profile dict from load_profile().
        jd_text: Optional job description text; banned words that appear in the JD are exempt.

    Returns:
        {"passed": bool, "errors": list[str], "warnings": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []
    mode = _normalize_validation_mode(validation_mode)

    # Required keys (no summary -- Jake's template doesn't have one)
    for key in ("title", "skills", "experience", "projects", "education"):
        if key not in data or not data[key]:
            errors.append(f"Missing required field: {key}")
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}

    # Collect all text for bulk checks
    all_text_parts: list[str] = []

    # Skills: check for fabrication (skip if skill is in the user's boundary or in the JD)
    allowed_skills = _build_skills_set(profile)
    jd_lower = jd_text.lower() if jd_text else ""
    if isinstance(data["skills"], dict):
        skills_text = " ".join(str(v) for v in data["skills"].values()).lower()
        for fake in FABRICATION_WATCHLIST:
            if len(fake) <= 2:
                continue
            if fake in skills_text:
                # Allow if the skill is in the user's skills boundary
                if any(fake in s for s in allowed_skills):
                    continue
                # Allow if the skill appears in the JD
                if fake in jd_lower:
                    continue
                errors.append(f"Fabricated skill: '{fake}'")

    # Experience: preserved companies must be present (Code Ninjas is optional)
    resume_facts = profile.get("resume_facts", {})
    preserved_companies = resume_facts.get("preserved_companies", [])

    if isinstance(data["experience"], list):
        for company in preserved_companies:
            # Code Ninjas can be dropped per user config
            if "code ninjas" in company.lower():
                continue
            has_company = any(
                company.lower() in str(e.get("company", e.get("header", ""))).lower()
                for e in data["experience"]
            )
            if not has_company:
                errors.append(f"Company '{company}' missing from experience")
        for entry in data["experience"]:
            for b in entry.get("bullets", []):
                all_text_parts.append(b)

    # Projects: collect bullets (projects may be fabricated/rewritten per user config)
    if isinstance(data["projects"], list):
        for entry in data["projects"]:
            for b in entry.get("bullets", []):
                all_text_parts.append(b)

    # Education: preserved school must be present
    preserved_school = resume_facts.get("preserved_school", "")
    if preserved_school:
        edu = data.get("education", "")
        # Education can be a dict (new schema) or a string (old schema)
        if isinstance(edu, dict):
            edu_text = " ".join(str(v) for v in edu.values()).lower()
        else:
            edu_text = str(edu).lower()
        if preserved_school.lower() not in edu_text:
            errors.append(f"Education '{preserved_school}' missing")

    # Bulk checks on all text (word-boundary matching)
    all_text = " ".join(all_text_parts).lower()

    found_banned = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", all_text)]
    # Exempt banned words that appear in the JD itself (mirroring JD language is ATS-optimal)
    if jd_text:
        jd_lower = jd_text.lower()
        found_banned = [w for w in found_banned if not re.search(r"\b" + re.escape(w) + r"\b", jd_lower)]
    if found_banned:
        banned_msg = f"Banned words: {', '.join(found_banned[:3])}"
        if mode == "strict":
            errors.append(banned_msg)
        elif mode == "normal":
            warnings.append(banned_msg)

    # Soft-banned words: warn but don't block
    found_soft = [w for w in SOFT_BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", all_text)]
    if jd_text:
        jd_lower = jd_text.lower() if not jd_text else jd_lower  # reuse if already computed
        found_soft = [w for w in found_soft if not re.search(r"\b" + re.escape(w) + r"\b", jd_lower)]
    if found_soft and mode in {"strict", "normal"}:
        warnings.append(f"Soft-banned words: {', '.join(found_soft[:3])}")

    found_leaks = [p for p in LLM_LEAK_PHRASES if p in all_text]
    if found_leaks:
        errors.append(f"LLM self-talk: '{found_leaks[0]}'")

    return {"passed": len(errors) == 0, "errors": errors, "warnings": warnings}


# ── Full Resume Text Validation ───────────────────────────────────────────

def validate_tailored_resume(text: str, profile: dict, original_text: str = "") -> dict:
    """Programmatic validation of a tailored resume against the user's profile.

    Args:
        text: The tailored resume text to validate.
        profile: User profile dict from load_profile().
        original_text: The original base resume text (for fabrication comparison).

    Returns:
        {"passed": bool, "errors": list[str], "warnings": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []
    text_lower = text.lower()

    personal = profile.get("personal", {})
    resume_facts = profile.get("resume_facts", {})

    # 1. Check required sections exist (flexible matching)
    section_variants: dict[str, list[str]] = {
        "SUMMARY": ["summary", "professional summary", "profile"],
        "TECHNICAL SKILLS": ["technical skills", "skills", "tech stack", "core skills", "technologies"],
        "EXPERIENCE": ["experience", "work experience", "professional experience"],
        "PROJECTS": ["projects", "personal projects", "key projects", "selected projects"],
        "EDUCATION": ["education", "academic background"],
    }
    for section, variants in section_variants.items():
        if not any(v in text_lower for v in variants):
            errors.append(f"Missing required section: {section} (or variant)")

    # 2. Check name preserved (warn, don't error -- we can inject it)
    full_name = personal.get("full_name", "")
    if full_name and full_name.lower() not in text_lower:
        warnings.append(f"Name '{full_name}' missing -- will be injected")

    # 3. Check companies preserved
    for company in resume_facts.get("preserved_companies", []):
        if company.lower() not in text_lower:
            errors.append(f"Company '{company}' missing -- cannot remove real experience")

    # 4. Projects: allowed to be fabricated/rewritten per user config
    # preserved_projects is intentionally empty when user allows fabrication
    for project in resume_facts.get("preserved_projects", []):
        if project.lower() not in text_lower:
            pass  # projects can be freely changed/fabricated

    # 5. Check school preserved
    preserved_school = resume_facts.get("preserved_school", "")
    if preserved_school and preserved_school.lower() not in text_lower:
        errors.append(f"Education '{preserved_school}' missing")

    # 6. Check contact info preserved (warn, don't error -- we can inject)
    email = personal.get("email", "")
    phone = personal.get("phone", "")
    if email and email.lower() not in text_lower:
        warnings.append("Email missing -- will be injected")
    if phone and phone not in text:
        warnings.append("Phone missing -- will be injected")

    # 7. Scan TECHNICAL SKILLS section for fabricated tools
    skills_start = text_lower.find("technical skills")
    skills_end = text_lower.find("experience", skills_start) if skills_start != -1 else -1
    if skills_start != -1 and skills_end != -1:
        skills_block = text_lower[skills_start:skills_end]
        for fake in FABRICATION_WATCHLIST:
            if len(fake) <= 2:
                continue
            if fake in skills_block:
                errors.append(f"FABRICATED SKILL in Technical Skills: '{fake}'")

    # 8. Scan full document for fabrication watchlist items not in original
    if original_text:
        original_lower = original_text.lower()
        for fake in FABRICATION_WATCHLIST:
            if len(fake) <= 2:
                continue
            if fake in text_lower and fake not in original_lower:
                warnings.append(f"New tool/skill appeared: '{fake}' (not in original)")

    # 9. Em dashes (should be auto-fixed by sanitize_text, but safety net)
    if "\u2014" in text or "\u2013" in text:
        errors.append("Contains em dash or en dash.")

    # 10. Banned words (word-boundary matching)
    found_banned = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text_lower)]
    if found_banned:
        errors.append(f"Banned words: {', '.join(found_banned[:5])}")

    # 11. LLM self-talk leak detection
    found_leaks = [p for p in LLM_LEAK_PHRASES if p in text_lower]
    if found_leaks:
        errors.append(f"LLM self-talk: '{found_leaks[0]}'")

    # 12. Duplicate section detection
    for section_name in ["summary", "experience", "education", "projects"]:
        count = text_lower.count(f"\n{section_name}\n") + text_lower.count(f"\n{section_name} \n")
        if text_lower.startswith(f"{section_name}\n"):
            count += 1
        if count > 1:
            errors.append(f"Section '{section_name}' appears {count} times.")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ── Cover Letter Validation ──────────────────────────────────────────────

def validate_cover_letter(text: str, validation_mode: str = "normal") -> dict:
    """Programmatic validation of a cover letter.

    Args:
        text: The cover letter text to validate.
        validation_mode: One of strict, normal, lenient.

    Returns:
        {"passed": bool, "errors": list[str], "warnings": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []
    mode = _normalize_validation_mode(validation_mode)
    text_lower = text.lower()

    # 1. Em dashes
    if "\u2014" in text or "\u2013" in text:
        errors.append("Contains em dash or en dash.")

    # 2. Banned words (word-boundary matching)
    found = [w for w in BANNED_WORDS if re.search(r"\b" + re.escape(w) + r"\b", text_lower)]
    if found:
        banned_msg = f"Banned words: {', '.join(found[:5])}"
        if mode == "strict":
            errors.append(banned_msg)
        elif mode == "normal":
            warnings.append(banned_msg)

    # 3. Too long
    words = len(text.split())
    if words > 300:
        errors.append(f"Too long ({words} words). Max 250.")

    # 4. LLM self-talk
    found_leaks = [p for p in LLM_LEAK_PHRASES if p in text_lower]
    if found_leaks:
        errors.append(f"LLM self-talk: '{found_leaks[0]}'")

    # 5. Must start with "Dear"
    stripped = text.strip()
    if not stripped.lower().startswith("dear"):
        errors.append("Must start with 'Dear Hiring Manager,'")

    return {"passed": len(errors) == 0, "errors": errors, "warnings": warnings}


# ── ATS Compliance Validation (Steps 1 & 2) ──────────────────────────────

# Common filler/non-technical words to ignore when extracting JD keywords
_FILLER = {
    "the", "and", "for", "with", "you", "your", "our", "that", "this", "will",
    "are", "from", "have", "has", "been", "their", "they", "what", "about",
    "work", "team", "role", "join", "must", "need", "also", "can", "may",
    "should", "would", "could", "into", "such", "like", "than", "more",
    "other", "each", "all", "any", "both", "through", "between", "over",
    "after", "before", "during", "under", "above", "across", "within",
    "including", "experience", "years", "strong", "ability", "skills",
    "knowledge", "understanding", "proficiency", "excellent", "good",
    "working", "using", "development", "develop", "design", "build",
    "create", "manage", "support", "ensure", "provide", "maintain",
    "review", "implement", "deliver", "drive", "lead", "collaborate",
    "communicate", "analyze", "test", "write", "document", "troubleshoot",
    "apply", "looking", "seeking", "ideal", "candidate", "requirements",
    "qualifications", "responsibilities", "preferred", "required", "minimum",
    "plus", "bonus", "nice", "equivalent", "related", "relevant", "similar",
    "based", "hands-on", "proven", "deep", "high", "level", "senior",
    "junior", "intermediate", "advanced", "expert", "proficient",
    "environment", "environments", "systems", "system", "tools", "tool",
    "technologies", "technology", "platforms", "platform", "solutions",
    "solution", "services", "service", "applications", "application",
    "software", "data", "code", "processes", "process",
    "company", "business", "client", "clients", "customer", "customers",
    "performance", "quality", "security", "scalable", "reliable",
    "location", "remote", "hybrid", "onsite", "full-time", "part-time",
    "contract", "permanent", "salary", "benefits", "compensation",
    "equal", "opportunity", "employer", "diversity", "inclusion",
    # Noise from JD boilerplate
    "office", "street", "tower", "bay", "floor", "suite", "building",
    "city", "town", "village", "avenue", "boulevard", "road", "drive",
    "north", "south", "east", "west", "downtown", "midtown", "uptown",
    "world", "global", "international", "national", "local", "regional",
    "mutual", "prosper", "communities", "thrive", "growing", "challenge",
    "progressive", "dynamic", "collaborative", "inclusive", "respectful",
    "legal", "admin", "accommodation", "barrier", "proud", "committed",
    "initiatives", "organization", "advisory", "coaching", "training",
    "commissions", "stock", "bonuses", "flexible", "competitive",
    "chief", "officer", "director", "manager", "president", "vice",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "hours", "week", "month", "year", "day", "time", "date",
    "toronto", "vancouver", "montreal", "ottawa", "calgary", "edmonton",
    "new", "york", "francisco", "angeles", "chicago", "boston", "seattle",
    "canada", "ontario", "british", "columbia", "alberta", "quebec",
    "please", "visit", "learn", "click", "here", "more", "details",
    "vacancy", "posting", "posted", "closes", "deadline",
    # Common JD boilerplate words that look like acronyms or tech
    "pm", "am", "st", "nd", "rd", "th", "hr", "eg", "ie", "vs",
    "grp", "cdo", "sdmo", "rbc", "elk", "llc", "inc", "ltd", "corp",
    "debt", "vault", "load", "output", "parties", "conditions",
    "documents", "milestones", "candidates", "strategies",
    "integrity", "reliability", "availability", "latency",
    "optimization", "remediation", "detail", "detail-oriented",
    "pipelines", "pipeline", "ai-assisted", "ai-enabled", "e.g", "i.e",
    "ll-post", "sdlc",
    # Noisy English words that pass uppercase heuristics
    "patterns", "practices", "expertise", "net",
    # Short all-caps that aren't tech: company names, hashtags, common abbreviations
    "cgi", "ia", "it", "us", "li-bn", "bn", "li", "ci", "cd",
    "rbc", "td", "bmo", "cibc", "ibm",
    # Roman numerals, common 2-letter words that appear uppercase in JDs
    "ii", "iii", "iv", "vi", "on", "no", "or", "an", "at", "to", "do",
    "of", "in", "so", "up", "if", "be", "my", "we", "he", "me",
    # Timezone/location/social abbreviations
    "est", "pst", "cst", "mst", "nyc", "sf", "la", "dc",
    "linkedin", "glassdoor", "indeed",
    # Non-tech concepts
    "dsa", "ada", "problem-solving",
}

# Known single-word tech terms that don't trigger heuristics
# (title-cased words like React, Vue whose word[1:] is all lowercase)
_KNOWN_TECH_WORDS = {
    # Languages
    "python", "java", "javascript", "typescript", "golang", "ruby", "rust",
    "kotlin", "swift", "scala", "perl", "elixir", "clojure", "haskell",
    # Frontend
    "react", "vue", "angular", "svelte", "ember", "jquery",
    # Backend
    "django", "flask", "fastapi", "express", "nestjs", "rails", "laravel",
    "spring", "hibernate", "quarkus", "micronaut",
    # Databases
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "couchbase", "sqlite", "mssql", "mariadb",
    # Cloud/DevOps
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "gitlab",
    "datadog", "grafana", "prometheus", "nginx",
    # Messaging/Data
    "kafka", "rabbitmq", "celery", "airflow", "spark", "flink",
    "snowflake", "databricks", "dbt", "hadoop", "hive", "presto",
    # Tools
    "graphql", "grpc", "websocket", "oauth", "saml",
    "git", "maven", "gradle", "webpack", "vite",
    "jest", "pytest", "selenium", "cypress", "playwright",
    "figma", "tableau", "looker",
    # Mobile
    "flutter", "xamarin",
    # AI/ML
    "tensorflow", "pytorch", "keras", "pandas", "numpy", "opencv", "langchain",
}

# Known technical terms (multi-word) that should be matched as phrases
_KNOWN_TECH_PHRASES = [
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data science", "data engineering", "data pipeline",
    "data pipelines", "data governance", "data lineage", "data modeling",
    "data quality", "data mesh", "data lake", "data lakehouse", "data warehouse",
    "row level security", "code review", "code reviews", "test plans",
    "defect tracking", "defect reporting", "unit testing", "integration testing",
    "regression testing", "performance testing", "manual testing", "e2e testing",
    "ci/cd", "ci cd", "azure devops", "spring boot", "spring framework",
    ".net core", "react.js", "react hooks", "context api", "node.js",
    "vue.js", "angular.js", "next.js", "express.js",
    "rest api", "restful api", "restful apis", "graphql",
    "web3.js", "web3",
    "amazon web services", "google cloud", "google cloud platform",
    "apache kafka", "apache flink", "apache spark", "apache airflow",
    "aws lambda", "aws ecs", "aws eks", "aws s3", "aws glue",
    "azure functions", "azure devops",
    "terraform", "infrastructure as code",
    "docker compose", "github actions",
    "agile methodology", "agile/scrum", "scrum master",
    "full stack", "front end", "front-end", "back end", "back-end",
    "soc2", "soc 2", "gdpr", "hipaa", "owasp",
]


def _extract_jd_keywords(jd_text: str) -> set[str]:
    """Extract technical keywords and phrases from a job description.

    Returns a set of lowercase keywords/phrases that the resume should contain.
    """
    jd_lower = jd_text.lower()
    keywords: set[str] = set()

    # 1. Match known multi-word tech phrases
    for phrase in _KNOWN_TECH_PHRASES:
        if phrase in jd_lower:
            keywords.add(phrase)

    # 2. Extract words that look technical (contain mixed case, digits, dots,
    #    slashes, or are short all-caps acronyms in the original text)
    #    Note: "/" excluded from regex — slash terms (ci/cd etc.) are handled
    #    by _KNOWN_TECH_PHRASES in step 1 above.
    words = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*", jd_text)
    for word in words:
        # Strip trailing punctuation BEFORE heuristic check — prevents
        # "opportunities." from triggering the special-char heuristic due to "."
        word = word.rstrip(".,;:!?()")
        w_lower = word.lower()
        if len(w_lower) < 2 or w_lower in _FILLER:
            continue
        # Skip URLs, emails, and paths
        if "." in w_lower and any(w_lower.endswith(ext) for ext in [".com", ".ca", ".org", ".net", ".io", ".gov"]):
            continue
        # Skip word-digit patterns like "Hybrid-2" (from "Hybrid - 2 days")
        if re.match(r"^[a-z]+-\d+$", w_lower):
            continue
        is_tech = (
            # Mixed case within word: React, PostgreSQL, JavaScript, FastAPI
            (any(c.isupper() for c in word[1:]) and any(c.islower() for c in word))
            # Has digits: S3, EC2, Web3, H2O
            or any(c.isdigit() for c in word)
            # Has special chars (interior only): C#, C++
            or any(c in "#+" for c in word)
            # Short all-caps: SQL, AWS, GCP, API, ETL, ML, AI, ELK
            or (len(word) <= 5 and word.isupper() and len(word) >= 2)
            # Known tech term (catches React, Vue, Docker, etc.)
            or w_lower in _KNOWN_TECH_WORDS
        )
        if is_tech:
            keywords.add(w_lower)

    # 3. For hyphenated keywords, also add the tech-looking part
    #    e.g. "mssql-backed" → add "mssql" (the known tech part)
    hyphenated = [kw for kw in keywords if "-" in kw and " " not in kw]
    for kw in hyphenated:
        parts = kw.split("-")
        for part in parts:
            if part in _KNOWN_TECH_WORDS or part.isupper():
                keywords.add(part)
        keywords.discard(kw)

    # 4. Deduplicate: remove single words already covered by phrases
    #    e.g. if "spring boot" is present, don't also require "spring"
    to_remove = set()
    for kw in keywords:
        if " " not in kw:
            for phrase in keywords:
                if " " in phrase and kw in phrase.split():
                    to_remove.add(kw)
                    break
    # Also remove variant forms: "springboot" when "spring boot" exists,
    # "react.js" when "react" exists, etc.
    for kw in list(keywords):
        nospace = kw.replace(" ", "")
        nodot = kw.replace(".js", "").replace(".ts", "")
        for other in keywords:
            if other == kw:
                continue
            if nospace == other or other.replace(" ", "") == kw:
                to_remove.add(max(kw, other, key=len))  # keep shorter
            if nodot == other or other.replace(".js", "").replace(".ts", "") == kw:
                to_remove.add(max(kw, other, key=len))  # keep shorter
    keywords -= to_remove

    # Remove certifications -- these can't be fabricated so shouldn't count
    # against keyword coverage
    cert_terms = {"cisa", "cissp", "pmp", "cka", "ckad", "aws certified",
                  "azure certified", "gcp certified", "scrum master",
                  "certified", "certification"}
    keywords -= cert_terms

    return keywords


def validate_ats_compliance(data: dict, jd_text: str) -> dict:
    """Validate the tailored resume JSON against ATS compliance steps 1 & 2.

    Step 1: Keyword coverage -- checks that 85%+ of JD technical keywords
    appear somewhere in the resume output (skills, bullets, skills_used).

    Step 2: Bullet formula -- checks that experience/project bullets contain
    the required "by" and "resulting in" structural markers.

    Args:
        data: Parsed JSON resume from the LLM.
        jd_text: Full job description text.

    Returns:
        {"passed": bool, "errors": list[str], "warnings": list[str],
         "keyword_coverage": float, "bullet_compliance": float,
         "missing_keywords": list[str], "bad_bullets": list[str]}
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not jd_text:
        return {"passed": True, "errors": [], "warnings": ["No JD text for ATS check"],
                "keyword_coverage": 0.0, "bullet_compliance": 0.0,
                "missing_keywords": [], "bad_bullets": []}

    # ── Step 1: Keyword Coverage ──
    jd_keywords = _extract_jd_keywords(jd_text)
    if not jd_keywords:
        return {"passed": True, "errors": [], "warnings": ["No keywords extracted from JD"],
                "keyword_coverage": 1.0, "bullet_compliance": 1.0,
                "missing_keywords": [], "bad_bullets": []}

    # Build the full text of the resume output for matching
    resume_parts: list[str] = []

    # Skills section
    skills = data.get("skills", {})
    if isinstance(skills, dict):
        for v in skills.values():
            resume_parts.append(str(v))

    # Experience bullets + skills_used
    for entry in data.get("experience", []):
        for b in entry.get("bullets", []):
            resume_parts.append(b)
        resume_parts.append(entry.get("skills_used", ""))
        resume_parts.append(entry.get("title", ""))

    # Project bullets + skills_used + tech_stack
    for entry in data.get("projects", []):
        for b in entry.get("bullets", []):
            resume_parts.append(b)
        resume_parts.append(entry.get("skills_used", ""))
        resume_parts.append(entry.get("tech_stack", ""))
        resume_parts.append(entry.get("name", ""))

    # Education coursework
    edu = data.get("education", {})
    if isinstance(edu, dict):
        resume_parts.append(edu.get("coursework", ""))

    resume_text_lower = " ".join(resume_parts).lower()
    # Un-escape common LaTeX sequences so keywords match
    resume_text_lower = (resume_text_lower
        .replace("\\#", "#").replace("\\%", "%")
        .replace("\\&", "&").replace("\\_", "_")
        .replace("\\textbackslash{}", "\\"))

    # Check each keyword
    missing: list[str] = []
    for kw in sorted(jd_keywords):
        # Flexible matching: "ci/cd" also matches "ci cd" and vice versa
        variants = [kw, kw.replace("/", " "), kw.replace(" ", "/"),
                     kw.replace("-", " "), kw.replace(" ", "-"),
                     kw.replace(".", ""), kw.replace(".js", "")]
        found = any(v in resume_text_lower for v in variants)
        if not found:
            missing.append(kw)

    coverage = 1.0 - (len(missing) / len(jd_keywords)) if jd_keywords else 1.0

    if coverage < 0.85:
        errors.append(
            f"Keyword coverage {coverage:.0%} < 85%. Missing: {', '.join(missing[:8])}"
        )
    elif missing:
        warnings.append(
            f"Keyword coverage {coverage:.0%}. Missing: {', '.join(missing[:5])}"
        )

    # ── Step 2: Bullet Formula ──
    all_bullets: list[str] = []
    for entry in data.get("experience", []):
        all_bullets.extend(entry.get("bullets", []))
    for entry in data.get("projects", []):
        all_bullets.extend(entry.get("bullets", []))

    bad_bullets: list[str] = []
    for bullet in all_bullets:
        b_lower = bullet.lower()
        has_by = " by " in b_lower
        has_resulting = "resulting in" in b_lower
        if not has_by or not has_resulting:
            missing_parts = []
            if not has_by:
                missing_parts.append('"by [action]"')
            if not has_resulting:
                missing_parts.append('"resulting in [impact]"')
            bad_bullets.append(
                f"Missing {' and '.join(missing_parts)}: \"{bullet[:80]}...\""
            )

    bullet_compliance = 1.0 - (len(bad_bullets) / len(all_bullets)) if all_bullets else 1.0

    if bad_bullets:
        errors.append(
            f"Bullet formula: {len(bad_bullets)}/{len(all_bullets)} bullets missing structure. "
            f"EVERY bullet MUST follow: '[Result verb] [metric] by [action], resulting in [impact]'. "
            f"Fix these: {bad_bullets[0]}"
        )

    # ── Step 3: Bullet Quality Checks ──
    import re as _re

    # 3a: Line overflow — bullets over 135 chars will wrap in Jake's template
    long_bullets: list[str] = []
    for bullet in all_bullets:
        if len(bullet) > 135:
            long_bullets.append(f'"{bullet[:60]}..." ({len(bullet)} chars)')
    if long_bullets:
        errors.append(
            f"Line overflow: {len(long_bullets)} bullets exceed 135 chars and will wrap. "
            f"Shorten each to under 130 chars to avoid orphan words on a 2nd line. Fix: {long_bullets[0]}"
        )

    # 3b: Tech dump at end — bullets ending with comma-separated tech lists
    tech_dump_re = _re.compile(
        r'(?:using|with|in|via|leveraging|utilizing)\s+'
        r'(?:[A-Z][A-Za-z0-9.#+/]*(?:,\s*| and ))+[A-Z][A-Za-z0-9.#+/]*\s*$'
    )
    tech_dump_bullets: list[str] = []
    for bullet in all_bullets:
        if tech_dump_re.search(bullet):
            tech_dump_bullets.append(f'"{bullet[:80]}..."')
    if tech_dump_bullets:
        errors.append(
            f"Tech dump: {len(tech_dump_bullets)} bullets end with a technology list instead of impact. "
            f"Move tools to skills_used, end bullets with measurable results. Fix: {tech_dump_bullets[0]}"
        )

    # 3c: Missing metrics — every bullet should contain at least one number
    no_metric_re = _re.compile(r'\d')
    no_metric_bullets: list[str] = []
    for bullet in all_bullets:
        if not no_metric_re.search(bullet):
            no_metric_bullets.append(f'"{bullet[:80]}..."')
    if no_metric_bullets:
        errors.append(
            f"No metrics: {len(no_metric_bullets)}/{len(all_bullets)} bullets contain zero numbers. "
            f"Every bullet MUST have a concrete metric (%, count, $, timeframe). "
            f"Fix: {no_metric_bullets[0]}"
        )

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "keyword_coverage": coverage,
        "bullet_compliance": bullet_compliance,
        "missing_keywords": missing,
        "bad_bullets": bad_bullets,
    }
