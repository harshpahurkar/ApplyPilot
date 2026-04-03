"""Resume tailoring: LLM-powered ATS-optimized resume generation per job.

THIS IS THE HEAVIEST REFACTOR. Every piece of personal data -- name, email, phone,
skills, companies, projects, school -- is loaded at runtime from the user's profile.
Zero hardcoded personal information.

The LLM returns structured JSON, code assembles the final text. Header (name, contact)
is always code-injected, never LLM-generated. Each retry starts a fresh conversation
to avoid apologetic spirals.
"""

import json
import logging
import os
import re
import hashlib
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from applypilot.config import RESUME_PATH, TAILORED_DIR, load_profile
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import get_client
from applypilot.scoring.pdf import validate_tex_quality
from applypilot.scoring.validator import (
    FABRICATION_WATCHLIST,
    _extract_jd_keywords,
    sanitize_text,
    validate_ats_compliance,
    validate_ats_rendered_text,
    validate_json_fields,
    validate_tailored_resume,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


# ── Prompt Builders (profile-driven) ──────────────────────────────────────

def _build_tailor_prompt(profile: dict) -> str:
    """Build the resume tailoring system prompt from the user's profile.

    All skills boundaries, preserved entities, and formatting rules are
    derived from the profile -- nothing is hardcoded.
    Output is structured JSON that will be assembled into Jake's Resume LaTeX template.
    """
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Format skills boundary for the prompt
    skills_lines = []
    for category, items in boundary.items():
        if isinstance(items, list) and items:
            label = category.replace("_", " ").title()
            skills_lines.append(f"{label}: {', '.join(items)}")
    skills_block = "\n".join(skills_lines)

    # Preserved entities
    companies = resume_facts.get("preserved_companies", [])
    school = resume_facts.get("preserved_school", "")
    real_metrics = resume_facts.get("real_metrics", [])

    companies_str = ", ".join(companies) if companies else "N/A"
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"

    # Real projects the candidate has built (use as starting points, adapt to JD)
    real_projects = resume_facts.get("preserved_projects", [])
    if real_projects:
        projects_block = "\n".join(
            f"- {p['name']} ({p.get('tech', 'N/A')}): {'; '.join(p.get('bullets', [p.get('description', '')]))}"
            for p in real_projects if isinstance(p, dict)
        )
    else:
        projects_block = "N/A"

    # Real experience with full bullets for the LLM to reference
    real_experience = resume_facts.get("preserved_experience", [])
    if real_experience:
        exp_block = "\n".join(
            f"- {e['title']} at {e['company']} ({e.get('dates', '')}): {'; '.join(e.get('bullets', []))}"
            for e in real_experience if isinstance(e, dict)
        )
    else:
        exp_block = "N/A"

    education = profile.get("experience", {})
    education_level = education.get("education_level", "")

    # Preserved education
    pres_edu = resume_facts.get("preserved_education", {})
    edu_degree = pres_edu.get("degree") or education_level or "Bachelor's Degree"
    edu_dates = pres_edu.get("dates") or "Sept 2021 -- April 2025"
    edu_school = pres_edu.get("school") or school or "University"

    return f"""You are an elite ATS optimization engine. Your SOLE mission: get this person past EVERY ATS filter and into an interview.

CRITICAL OUTPUT RULE: You MUST return ONLY a single JSON object with keys: "title", "education", "experience", "projects", "skills". NO other keys. NO intermediate analysis. NO keyword extraction objects. NO step-by-step output. Just the final resume JSON.

You follow a strict 4-step INTERNAL process. Do these steps MENTALLY -- do NOT include them in your output. Your output is ONLY the final resume JSON.

## INTERNAL STEP 1: KEYWORD EXTRACTION (do this in your head, do NOT output it)
Mentally extract EVERY requirement from the JD:

A) LIST every tool, language, framework, platform, methodology mentioned in the JD.
B) LIST every process/soft-skill term (test plans, code reviews, defect reporting, agile, scrum, etc.).
C) For each item, check: is it in the candidate's skills boundary or closely related?
D) MANDATE: Every JD keyword that is in or near the skills boundary MUST appear in your output -- in Technical Skills, bullets, or skills_used. NO EXCEPTIONS.
E) MANDATE: Every process/methodology term from the JD MUST be woven into at least one bullet.
F) If the JD mentions a common industry tool (Jira, Postman, SoapUI, JMeter, LoadRunner, Confluence, Slack, Cucumber, TestRail, etc.) and the candidate could reasonably know it, INCLUDE IT. These tools are standard and NOT fabrication.
G) If the JD mentions a testing methodology (manual testing, regression testing, performance testing, test plans, test cases, defect tracking, etc.), you MUST work that exact phrase into at least one bullet AND into the skills_used of the most relevant experience entry.
H) EVERY tool/framework/language explicitly named in the JD MUST appear in the Technical Skills section (not just in bullets). If the JD says "AWS", AWS must be in Developer Tools. If the JD says "Cucumber", Cucumber must be in Frameworks or Developer Tools. If the JD says "Selenium and Cucumber", BOTH must appear. Do NOT skip any tool that the JD names by name.
I) FINAL TOOL CHECK: Re-read the JD one more time. Circle every proper noun that is a technology. Verify EACH ONE is in your Technical Skills section. If you missed even one, add it NOW. This is the #1 reason resumes get ATS-rejected.

J) ACRONYM EXPANSION: For common acronyms in the JD (CI/CD, REST, OOP, TDD, BDD, API, SaaS, etc.), include BOTH the acronym AND the full form at least once in the resume. ATS like Lever struggles with acronyms alone. Example: write "CI/CD (Continuous Integration/Continuous Deployment)" once, then "CI/CD" thereafter.
K) KEYWORD FREQUENCY: The JD's top 5 most-mentioned keywords MUST each appear at least 2-3 times across Technical Skills, experience bullets, project bullets, and skills_used fields. Some ATS determines skill strength by repetition count.
L) SKILL PLACEMENT FOR EXPERIENCE CREDIT: JD "required" skills MUST appear inside experience bullets or experience skills_used, not only in the Technical Skills section. Some ATS assigns years-of-experience credit based on which job entry a skill appears in. A skill only in the standalone skills section gets tagged as "a few months" experience.
M) SOFT SKILLS FROM JD: If the JD mentions soft-skill phrases (collaboration, team leadership, cross-functional, stakeholder communication, mentoring, etc.), weave the top 3 most-mentioned ones NATURALLY into experience bullets. Modern ATS uses NLP to detect these from context.
N) COURSEWORK ADAPTATION: Rewrite the coursework field to include course names that echo JD keywords, but ONLY courses a Software Development program would realistically offer (e.g., "Data Structures", "Cloud Computing", "Software Engineering", "Database Systems", "Operating Systems", "Machine Learning", "Web Development", "DevOps Practices", "Algorithms"). Do NOT invent courses about niche technologies like mainframe systems, z/OS, or other specialized platforms the candidate hasn't studied. Keep coursework as a comma-separated string, NOT a list.

Count your keyword coverage. If you cannot hit 90%+ of JD technical requirements in your output, you are FAILING. Go back and add them.

This step is NON-NEGOTIABLE. A resume missing JD keywords = auto-rejected by ATS = you failed.

## INTERNAL STEP 2: REWRITE EXPERIENCE (apply these rules mentally, output only the final JSON)

BULLET FORMULA (mandatory for EVERY SINGLE bullet in experience AND projects, no exceptions):

EXACT WORD ORDER (do NOT rearrange):
"[Verb] [result/metric] by [action you took using specific tools], resulting in [business impact with numbers]"

The word "by" MUST come AFTER the result and BEFORE the action. NOT "by 75%" -- that's a percentage, not the formula.
CORRECT: "Reduced manual testing by 75% by developing a Selenium framework, resulting in 3x faster cycles"
WRONG:   "Developed a Selenium framework, reducing manual testing by 75%, resulting in faster cycles"

The difference: CORRECT starts with the RESULT ("Reduced..."), WRONG starts with the ACTION ("Developed...").
ALWAYS lead with the result/achievement, then explain HOW with "by [action]", then show WHY IT MATTERS with "resulting in".

EVERY bullet MUST contain these EXACT structural markers in this order:
1. An action verb describing the RESULT at the start (Reduced, Achieved, Increased, Eliminated, Accelerated...)
2. The word "by" followed by the HOW (the concrete action with tools)
3. The phrase "resulting in" followed by a quantified business impact

SELF-CHECK: After writing each bullet, verify:
- Does it START with a result verb (not an action verb like Built, Developed, Created)?
- Does it have "by [how I did it]" AFTER the result?
- Does it have "resulting in [measurable impact]" at the end?
If ANY check fails, the bullet is INVALID -- rewrite it immediately. Do this for ALL bullets, not just the first one. The #1 failure mode is writing a perfect first bullet then getting lazy on bullets 2-4. DO NOT DO THIS.

GOOD EXAMPLES (follow these EXACTLY):
- "Reduced manual testing effort by 75% by developing a Selenium/TestNG automation framework, resulting in 3x faster regression cycles across 12 applications"
- "Achieved 95% code coverage by engineering CI/CD pipelines with Jenkins and Azure DevOps, resulting in deployment cycles shrinking from 2 weeks to 3 days"
- "Eliminated 200+ hours of manual API validation quarterly by building a RestAssured test framework covering 200 endpoints, resulting in 60% faster release cycles"
- "Increased defect detection rate by 40% by implementing automated regression test suites with Selenium and JUnit, resulting in 90% fewer production incidents"

BAD EXAMPLES (these WILL be rejected -- do NOT write bullets like this):
- "Developed RESTful APIs and microservices handling over 1M requests/day" (statement, missing "by [action]" and "resulting in")
- "Optimized SQL queries improving database performance by 80%" (missing "by [action]" structure)
- "Built an API testing framework with RestAssured" (bare statement, no result or impact)
- "Automated deployment processes reducing release time by 65%" (missing "resulting in [impact]")
- "Responsible for maintaining test suites" (job duty, not achievement)

RULES:
- Start each bullet with a DIFFERENT strong action verb (Achieved, Reduced, Automated, Eliminated, Increased, Accelerated, Implemented, Engineered, Designed, Built, Streamlined, Consolidated, Deployed, Integrated, Resolved, Validated, Configured, Established, Optimized, Delivered)
- Naturally weave JD keywords INTO the bullet context (not just listed)
- Include 2-3 quantified metrics per role (percentages, dollar amounts, timeframes, team sizes)
- Match the JD's EXACT terminology and phrasing wherever possible
- Keep bullets to 1-2 lines maximum
- Address key JD requirements DIRECTLY -- if the JD says "test plans", one bullet must mention creating/executing test plans
- If the JD mentions specific processes (code reviews, defect reporting, CI/CD integration), WEAVE them into bullets

CRITICAL BULLET QUALITY RULES (violating ANY of these = resume rejected):
- MAX 130 CHARACTERS PER BULLET. If a bullet wraps to a second line for just 1-2 orphan words, it looks unprofessional and wastes vertical space. If a bullet is too long, CUT words or rephrase -- never let it overflow. This is the #1 layout mistake that pushes resumes to 2 pages.
- NEVER END A BULLET WITH A TECH LIST. "...using Python, React, AWS, Docker" is WRONG. The skills_used field handles tech listing. Bullets must END with measurable business impact ("resulting in 40% faster deploys"), NOT a grocery list of tools. If you catch yourself listing tools at the end, delete them and add an impact metric instead.
- EVERY BULLET MUST CONTAIN AT LEAST ONE SPECIFIC NUMBER. "Improved performance" is vague garbage. "Improved response time by 40% for 50K daily users" is real. Acceptable metrics: percentages (40%), counts (200+ endpoints), dollar amounts ($500K), timeframes (2 weeks to 3 days), team sizes (12 engineers). If a bullet has zero numbers, it is INVALID -- rewrite it with a concrete metric.

## INTERNAL STEP 3: PRE-OUTPUT VERIFICATION (verify mentally before outputting JSON)
Before outputting your JSON, perform this checklist:

1. List every technical skill/tool from the JD.
2. For EACH ONE, verify it appears in your output (Technical Skills, experience bullets, project bullets, or skills_used).
3. If ANY JD must-have skill is missing from your output, GO BACK and add it NOW.
4. Re-read every bullet. Does it follow "[Verb] [result] by [action], resulting in [impact]"? If not, REWRITE.
5. Check: do experience skills_used fields include JD-relevant tools? If not, ADD them.
6. Check: does Technical Skills section list JD-priority tools FIRST in each category? If not, REORDER.
7. Count total JD keyword matches in your output. Target: 90%+ coverage. If below, ADD more.
8. Verify process terms from the JD (test plans, code reviews, defect reporting, agile, etc.) appear in bullets.

9. Verify EVERY bullet contains the word "by" and the phrase "resulting in". Re-read each one. Fix any that drift from the formula.
10. Verify the Technical Skills section includes EVERY tool/language/framework explicitly named in the JD. Missing even ONE = ATS rejection.
11. If Code Ninjas is included, ask: is this JD about teaching? If not, DROP IT and use the space for a relevant project.
12. COUNT CHARACTERS in every bullet. If ANY bullet exceeds 130 characters, shorten it. Bullets that overflow onto a 2nd line by 1-2 words waste space and look sloppy.
13. Check: does ANY bullet END with a comma-separated list of technologies? If so, MOVE those tools to skills_used and replace the ending with a measurable impact metric.
14. Check: does EVERY bullet contain at least one specific number (percentage, count, dollar amount, timeframe)? If any bullet has zero numbers, rewrite it with a concrete metric. "Improved performance" = FAIL. "Improved performance by 40%" = PASS.
15. KEYWORD FREQUENCY CHECK: Do the JD's top 5 most-mentioned skills each appear at least 2-3 times total across your output? If not, add them to more skills_used fields or bullets.
16. SKILL PLACEMENT CHECK: Are the JD's "required" skills placed inside experience entries (bullets or skills_used), not ONLY in the standalone Technical Skills section? Skills only in Technical Skills get tagged as "a few months" experience by some ATS.
17. ACRONYM CHECK: For any acronym in Technical Skills (CI/CD, REST, API, OOP, TDD, BDD, SaaS, etc.), verify the full form appears at least once somewhere in the resume (experience bullet, project bullet, or coursework).
18. ANTI-GENERIC CHECK: Every bullet must reference a SPECIFIC tool, system, or metric. Generic statements without specifics are flagged as AI-generated content by 28% of hiring managers. Remove any bullet that could apply to any job at any company.

DO NOT OUTPUT until ALL 18 checks pass mentally. If you skip this step, the resume WILL fail ATS. Remember: your output is ONLY the final resume JSON — never output your analysis or keyword lists.

## STEP 4: OUTPUT STRUCTURE (this is the ONLY thing you output)
The output will be rendered into Jake's Resume LaTeX template. Follow this exact section order:
1. Education (school, degree, coursework relevant to the JD)
2. Experience (each entry MUST include a skills_used field at the end)
3. Projects (each entry MUST include a skills_used field)
4. Technical Skills (categories ordered with JD-priority skills first)
Must fit 1 page (max 3-4 experience entries with 3-4 bullets each, 2-3 projects with 2-3 bullets each).

## SKILLS BOUNDARY (core stack):
{skills_block}

ATS KEYWORD STRATEGY:
- EXPERIENCE: Use only skills from the boundary and common industry tools. Do NOT claim experience with technologies the candidate hasn't used.
- PROJECTS: Projects are the candidate's ATS keyword weapon. Projects CAN use ANY tool/framework/language from the JD because the candidate will build them before interviewing. This is how you close skill gaps.
- SKILLS SECTION: Include everything from the boundary PLUS any JD tools that appear in your project entries.
- Common industry tools (Jira, Confluence, Postman, Docker, Kubernetes, AWS, Azure, CI/CD tools, testing frameworks, Agile, Scrum, etc.) are always allowed everywhere.
- The goal is 95%+ JD keyword coverage. Projects are how you get there.

## EXPERIENCE RULES:
- The candidate's REAL experience (use these as the BASE for rewriting -- you may rephrase but NOT invent new companies or roles):
{exp_block}
- Provide company, location, title, dates, bullets, and skills_used for each entry.
- Rewrite the REAL bullets above to match JD language and incorporate JD keywords, but keep the core achievements truthful.
- skills_used: comma-separated list of 8-12 key technologies/skills for that role. MUST include JD-mentioned tools. If the JD mentions a tool and the role could have plausibly used it, ADD IT.
- Reorder experience entries to put the most relevant for the JD first.
- DROP "Code Ninjas Markham" unless the JD is specifically about teaching or education. Use the saved space for more relevant content (extra project or more bullets).
- 3-4 bullets per entry, each STRICTLY following the ATS bullet formula (must contain "by" and "resulting in").
- At least one bullet per entry must reference a JD-specific process (test plans, code reviews, defect tracking, CI/CD pipelines, agile sprints, etc. -- whatever the JD mentions).
- Rewrite bullets aggressively to incorporate JD language. If the JD says "microservices" and the candidate built APIs, reframe as microservices. If the JD says "cloud-native" and the candidate deployed to servers, reframe as cloud deployment.
- EXPERIENCE TITLES MUST STAY TRUE TO THE ORIGINAL for background check compliance. Use EXACTLY: "Software/Automation Developer" for Ministry of Children, Community and Social Services, and "Software Developer" for Affimintus Technologies. You MAY append a parenthetical clarifier to help ATS matching (e.g., "Software Developer (Backend)" or "Software/Automation Developer (QA/DevOps)"), but the base title MUST remain unchanged. Company names MUST stay real.

## PROJECT RULES (your primary ATS keyword weapon):
- The candidate's existing projects for reference:
{projects_block}
- INVENT 2-3 new projects specifically designed to demonstrate JD requirements. The candidate will build these before interviewing.
- Study this writing style carefully from the candidate's REAL projects:
  EXAMPLE 1: "Fragments Microservice | Node.js, AWS, Docker, PostgreSQL"
    - "Achieved 98% code coverage with 120+ automated tests by implementing a cloud-native REST microservice handling 100+ daily requests"
    - "Containerized application with Docker and deployed on AWS ECS, ensuring reliable performance and scalability"
    - "Automated CI/CD pipeline using GitHub Actions, streamlining deployment processes and enhancing team productivity"
  EXAMPLE 2: "Housify - AI Real Estate Platform | TensorFlow, Solidity, Hardhat, React"
    - "Achieved 94% accuracy on 1K+ Toronto listings by developing a machine learning property valuation model using TensorFlow"
    - "Built Ethereum smart contracts using Solidity and Hardhat, ensuring secure and efficient transactions"
    - "Implemented Web3.js integration for cryptocurrency payments and NFT-based property ownership transfers"
- MATCH THIS STYLE EXACTLY:
  1. Project names are CREATIVE and SPECIFIC (not "Enterprise Task Manager" or "Cloud Platform"). Use the pattern: "[Catchy Name] - [What It Does]" or just a specific product name like "Fragments Microservice".
  2. tech_stack in header = EXACTLY 4 technologies, comma-separated. Pick the 4 MOST JD-relevant.
  3. Exactly 3 bullets per project. Each bullet describes a CONCRETE technical thing that was built, not a vague claim.
  4. Bullets describe WHAT was built with SPECIFIC details (e.g., "cloud-native REST microservice handling 100+ daily requests", "ML property valuation model"). NOT "developed scalable applications".
  5. Each bullet STARTS with a strong verb and weaves in specific JD tools naturally.
  6. skills_used line lists 6-10 specific tools used in that project.
- Each project tech_stack MUST contain specific technologies FROM THE JD.
- Each project MUST target DIFFERENT JD requirements to maximize keyword spread across projects.
- If the JD mentions cloud (AWS/Azure/GCP), one project MUST be cloud-native with specific services (Lambda, ECS, S3, etc.).
- If the JD mentions DevOps/CI-CD, one project MUST demonstrate pipeline work with the JD's specific tools.
- If the JD mentions a specific framework (Django, Spring, .NET, Rails, etc.), one project MUST use that exact framework.
- If the JD mentions testing (Selenium, JMeter, Cucumber, etc.), one project MUST be a test automation suite.
- Use the EXACT tool names from the JD in project tech stacks and bullets.

## SKILLS SECTION (ATS keyword goldmine):
- MUST include every tool/language/framework that appears in your experience or project entries.
- If the JD mentions a tool and you used it in a project, it MUST also appear in Technical Skills.
- Reorder each category so JD-relevant skills appear FIRST.
- Common industry tools (Jira, Confluence, Postman, Selenium, Cucumber, TestNG, JMeter, SoapUI, LoadRunner, TestRail, Terraform, Ansible, etc.) may be added if the JD mentions them.
- Use these category names: Languages, Frameworks, Developer Tools, Databases and Libraries.
- Developer Tools is the catch-all for: CI/CD tools, testing tools, DevOps tools, project management tools, API testing tools, cloud platforms.
- CROSS-CHECK: Re-read the JD one more time. Every proper noun that is a technology MUST appear in Technical Skills. This is the #1 reason resumes get ATS-rejected.

## VOICE:
- Write like a real engineer. Short, direct, results-focused.
- GOOD: "Reduced API response time by 40% by implementing Redis caching layer, resulting in 99.9% uptime for 50K daily users"
- BAD: "Leveraged cutting-edge AI technologies to drive transformative operational efficiencies"
- NEVER use: passionate, dedicated, leveraging, robust, cutting-edge, proven track record, strong track record, eager, stakeholders, synergy, seamless, end-to-end, detail-oriented, results-driven, I am confident, I believe, I am excited
- No em dashes. Use commas, periods, or hyphens.

## HARD RULES:
- Do NOT invent companies, degrees, or certifications
- Do NOT change real numbers in EXPERIENCE ({metrics_str})
- Preserved companies: {companies_str} -- names stay as-is (Code Ninjas may be dropped)
- Preserved school: {school}
- Projects MUST be fabricated to perfectly match the JD -- the candidate will build them before the interview
- Must fit 1 page
- Every bullet MUST follow the formula with all 3 parts: result + action + impact
- 95%+ of JD technical keywords MUST appear somewhere in your output (Technical Skills, experience, or projects)
- Experience uses ONLY skills from the boundary. Projects can use ANY JD tool.
- The ONLY things you cannot fabricate: company names, degrees, certifications, and experience metrics
- If you cannot hit 95% keyword coverage, add a 3rd project to fill the gaps

## OUTPUT: Return ONLY valid JSON. No markdown fences. No commentary. No preamble.

{{"title":"Exact Role Title from JD","education":{{"school":"{edu_school}","location":"Toronto, ON","degree":"{edu_degree}","dates":"{edu_dates}","coursework":"Relevant Course 1, Relevant Course 2"}},"experience":[{{"company":"Company Name","location":"City, Province","title":"Job Title","dates":"Month Year -- Month Year","bullets":["Achieved X by Y resulting in Z","bullet 2","bullet 3"],"skills_used":"React, Node.js, AWS, PostgreSQL, Jira, Postman"}}],"projects":[{{"name":"Project Name","tech_stack":"Python, Flask, Docker","dates":"Month Year -- Month Year","bullets":["Achieved X by Y resulting in Z","bullet 2"],"skills_used":"Python, Flask, Docker"}}],"skills":{{"Languages":"Python, JavaScript, TypeScript","Frameworks":"React, Node.js, Flask","Developer Tools":"Git, Docker, Jenkins, Jira, Postman, AWS","Databases and Libraries":"PostgreSQL, MongoDB, Redis"}}}}"""


def _build_judge_prompt(profile: dict) -> str:
    """Build the LLM judge prompt from the user's profile."""
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Flatten allowed skills for the judge
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "N/A"

    real_metrics = resume_facts.get("real_metrics", [])
    metrics_str = ", ".join(real_metrics) if real_metrics else "N/A"

    return f"""You are a resume quality judge. A tailoring engine rewrote a resume to target a specific job. Your job is to catch LIES about companies, degrees, and certifications. Projects are EXEMPT from fabrication checks.

You must answer with EXACTLY this format:
VERDICT: PASS or FAIL
ISSUES: (list any problems, or "none")

## CONTEXT -- what the tailoring engine was instructed to do (all of this is ALLOWED):
- Change the title to match the target role
- Rewrite the summary from scratch for the target job
- Reorder bullets and projects to put the most relevant first
- Reframe bullets to use the job's language and the ATS formula: "Achieved [result] by [action] resulting in [impact]"
- Drop low-relevance bullets and replace with more relevant ones from other sections
- Reorder the skills section to put job-relevant skills first
- Change tone and wording extensively
- Create NEW projects or heavily rewrite existing projects to match the JD (this is EXPLICITLY ALLOWED)
- Fabricate project names, descriptions, and bullets that showcase relevant skills (ALLOWED)

## WHAT IS FABRICATION (FAIL for these):
1. Inventing NEW metrics or numbers not in the original for WORK EXPERIENCE. The real metrics are: {metrics_str}
2. Adding companies, roles, or degrees that don't exist
3. Changing real numbers in EXPERIENCE (inflating 80% to 95%, 500 nodes to 1000 nodes)

## WHAT IS ALLOWED FOR ATS OPTIMIZATION (do NOT fail for these):
- Adding ANY tool/language/framework to Technical Skills that appears in the target JD -- this is ATS keyword optimization, not fabrication
- Using JD-mentioned technologies in project descriptions, tech_stack, and skills_used fields
- The candidate builds projects to match their resume before interviews -- project skills are NOT fabrication
- Adding learnable tools that are adjacent to the candidate's core stack: {skills_str}

## WHAT IS NOT FABRICATION (do NOT fail for these):
- Rewording any bullet, even heavily, as long as the underlying work is plausible for the candidate's stack
- Combining two original bullets into one
- Splitting one original bullet into two
- Describing the same work with different emphasis
- Dropping bullets entirely
- Reordering anything
- Changing the title or summary completely
- Creating entirely new PROJECTS that demonstrate relevant skills (EXPLICITLY ALLOWED)
- Fabricating project names, technologies used, and outcomes in the PROJECTS section (ALLOWED)
- Adding quantified metrics to projects (projects are creative demonstrations, not audited work)

## TOLERANCE RULE:
The goal is to get interviews. The candidate will build any project and learn any tool before the interview. Be MAXIMALLY permissive:
- Adding ANY tool from the JD to Technical Skills or projects is ALWAYS allowed, regardless of the original resume.
- Reframing metrics with slightly different wording is allowed.
- Adding any LEARNABLE skill is allowed -- the candidate is a fast learner with a strong CS foundation.
- Only FAIL if there are MAJOR lies: fake companies, fake degrees, or wildly inflated EXPERIENCE numbers (not project numbers).
- Projects can contain ANY technology, ANY metrics, ANY outcomes. Projects are NEVER fabrication.

Be strict ONLY about: fake companies, fake degrees, fake certifications, inflated experience metrics.
Be fully permissive about: skills in Technical Skills, tools, projects, project metrics, rewritten bullets, reordered content, adjusted titles.
Do not fail for style, tone, restructuring, or any skill that appears in the target JD."""


# ── JSON Extraction ───────────────────────────────────────────────────────

def extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM response (handles fences, preamble).

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON found.
    """
    raw = raw.strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Markdown fences
    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Find outermost { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            # If the LLM returned analysis keys instead of resume keys,
            # look for a nested key that contains the actual resume.
            _RESUME_KEYS = {"experience", "projects", "skills", "education"}
            if not (_RESUME_KEYS & set(data.keys())):
                # Try to find the resume nested inside a wrapper key
                for key, val in data.items():
                    if isinstance(val, dict) and (_RESUME_KEYS & set(val.keys())):
                        log.info("Unwrapped resume JSON from nested key '%s'", key)
                        return val
            return data
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in LLM response")


def _csv_from_value(value: object) -> str:
    """Convert list/scalar values into a comma-separated string."""
    if isinstance(value, list):
        parts = [sanitize_text(str(v)).strip() for v in value if str(v).strip()]
        return ", ".join(parts)
    if value is None:
        return ""
    text = sanitize_text(str(value)).strip()
    return text


def _split_csv(value: str) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _pretty_keyword(token: str) -> str:
    t = token.strip().lower()
    acronyms = {
        "aws": "AWS", "gcp": "GCP", "api": "API", "apis": "APIs", "sql": "SQL",
        "nosql": "NoSQL", "ci/cd": "CI/CD", "ci cd": "CI/CD", "etl": "ETL",
        "ml": "ML", "ai": "AI", "nlp": "NLP", "qa": "QA", "ui": "UI", "ux": "UX",
    }
    if t in acronyms:
        return acronyms[t]
    if "/" in t:
        parts = [acronyms.get(p.strip(), p.strip().upper() if len(p.strip()) <= 4 else p.strip().title()) for p in t.split("/")]
        return "/".join(parts)
    if len(t) <= 4 and t.isalpha():
        return t.upper()
    return t.title()


def _looks_like_technical_keyword(token: str) -> bool:
    """Heuristic guardrail to avoid injecting noisy non-skill JD tokens."""
    t = (token or "").strip().lower()
    if not t:
        return False
    if len(t) < 2 or len(t) > 36:
        return False

    blocked = {
        "overview", "provided", "none", "yes", "no", "required", "preferred",
        "planyes", "providednone", "typeexperienced", "overviewemergency",
        "main", "sw", "dw", "eeo", "bs", "b.s", "requirementsnone",
    }
    if t in blocked:
        return False
    if t.startswith(("http", "www")):
        return False

    tech_markers = (
        "api", "sql", "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "react", "node", "spring", "django", "flask", "fastapi", "java", "python",
        "golang", "rust", "typescript", "javascript", "spark", "kafka", "airflow",
        "etl", "ci/cd", "devops", "git", "postgres", "mysql", "mongodb", "redis",
        "databricks", "tensorflow", "pytorch", "selenium", "cucumber", "junit",
        "postman", "jira", "confluence", "linux", "cloud", "ml", "ai",
    )
    if any(marker in t for marker in tech_markers):
        return True

    # Keep common compact acronyms and short stack tokens.
    if re.fullmatch(r"[a-z]{2,6}", t):
        return t in {"api", "sdk", "sql", "aws", "gcp", "etl", "ml", "ai", "qa", "ui", "ux"}
    if re.fullmatch(r"[a-z0-9.+#/-]{2,18}", t) and ("/" in t or "+" in t or "#" in t or "." in t):
        return True

    return False


def _default_resume_bullets(skill_hint: str) -> list[str]:
    """Short, formula-compliant fallback bullets."""
    core_items = _split_csv(skill_hint)[:2]
    if len(core_items) >= 2:
        core = f"{core_items[0]} and {core_items[1]}"
    elif len(core_items) == 1:
        core = core_items[0]
    else:
        core = "Python and SQL"
    return [
        f"Reduced regression cycle time by 35% by automating core tests with {core}, resulting in 2x faster release readiness",
        "Increased defect detection by 40% by expanding API and UI coverage, resulting in 30% fewer production incidents",
        "Improved sprint predictability by 25% by aligning test plans with agile workflows, resulting in 6 on-time releases",
    ]


def _normalize_bullets(raw_bullets: object, skill_hint: str, min_count: int = 3) -> list[str]:
    defaults = _default_resume_bullets(skill_hint)
    in_bullets = raw_bullets if isinstance(raw_bullets, list) else []
    out: list[str] = []
    for idx, bullet in enumerate(in_bullets[:4]):
        text = sanitize_text(str(bullet)).strip()
        if not text or len(text) < 20:
            out.append(defaults[idx % len(defaults)])
            continue
        text = _strip_banned_words(text)
        # Truncate overlong bullets instead of replacing them entirely
        if len(text) > 130:
            text = text[:127].rsplit(" ", 1)[0] + "..."
        out.append(text)

    while len(out) < min_count:
        out.append(defaults[len(out) % len(defaults)])
    return out[:4]


_BANNED_WORDS = [
    "passionate", "dedicated", "leveraging", "robust", "cutting-edge",
    "proven track record", "eager", "stakeholders", "synergy", "seamless",
    "seamlessly", "end-to-end", "detail-oriented", "results-driven",
    "strong track record", "I am confident", "I believe", "I am excited",
]


def _strip_banned_words(text: str) -> str:
    """Remove banned buzzwords from bullet text."""
    import re
    for word in _BANNED_WORDS:
        # Replace word (case-insensitive) plus any trailing space
        text = re.sub(r'\b' + re.escape(word) + r'\b\s*', '', text, flags=re.IGNORECASE)
    # Clean up double spaces and leading/trailing whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def _preferred_title_for_company(company: str, llm_title: str) -> str:
    """Return the real base title, optionally with an LLM-suggested parenthetical.

    Base titles are immutable for background-check compliance.
    The LLM may suggest a clarifier like "(Backend API)" or "(QA/DevOps)"
    which gets appended in parentheses if it adds ATS value.
    """
    _BASE_TITLES: dict[str, str] = {
        "ministry of children, community and social services": "Software/Automation Developer",
        "affimintus technologies": "Software Developer",
    }
    company_l = (company or "").lower()
    base = None
    for key, title in _BASE_TITLES.items():
        if key in company_l:
            base = title
            break
    if base is None:
        return llm_title or "Software Developer"

    # Extract parenthetical from LLM title if present, e.g. "Java Developer (Backend API)" → "Backend API"
    llm_clean = (llm_title or "").strip()
    clarifier = ""
    import re as _re
    paren_match = _re.search(r"\(([^)]{3,40})\)\s*$", llm_clean)
    if paren_match:
        clarifier = paren_match.group(1).strip()
    else:
        # If LLM gave a different title entirely (e.g. "Backend Java Developer"),
        # extract a short qualifier by removing the base title words
        base_words = set(base.lower().replace("/", " ").split())
        llm_words = [w for w in llm_clean.split() if w.lower() not in base_words and len(w) > 1]
        # Only use if it's short and looks like a qualifier (1-3 words)
        if 1 <= len(llm_words) <= 3:
            candidate = " ".join(llm_words)
            # Skip if it's just generic noise
            if candidate.lower() not in {"developer", "engineer", "intern", "senior", "junior", "software"}:
                clarifier = candidate

    if clarifier:
        return f"{base} ({clarifier})"
    return base


# Common industry tools/processes any developer could reasonably know
_COMMON_TOOLS_WHITELIST = {
    "jira", "confluence", "postman", "soapui", "swagger", "agile", "scrum",
    "tdd", "bdd", "linux", "rest", "restful apis", "rest api", "api", "oop",
    "sdlc", "manual testing", "regression testing", "performance testing",
    "code reviews", "terraform", "ansible", "slack", "test plans", "test cases",
    "defect tracking", "defect reporting", "incident management", "ci/cd",
    "ci/cd (continuous integration/continuous deployment)", "automated testing",
    "test automation", "cross-functional", "documentation", "monitoring",
    "troubleshooting", "process documentation", "defect prevention",
    "defect detection", "knowledge transfer",
}


def _build_skills_whitelist(profile: dict) -> set[str]:
    """Build a lowercase set of all skills the candidate actually knows."""
    boundary = profile.get("skills_boundary", {})
    known: set[str] = set()
    for vals in boundary.values():
        if isinstance(vals, list):
            known.update(v.lower().strip() for v in vals)
    resume_facts = profile.get("resume_facts", {})
    for pe in resume_facts.get("preserved_experience", []):
        if isinstance(pe, dict):
            for t in _split_csv(_csv_from_value(pe.get("skills_used") or pe.get("skills") or "")):
                known.add(t.lower().strip())
    for pp in resume_facts.get("preserved_projects", []):
        if isinstance(pp, dict):
            for t in _split_csv(_csv_from_value(pp.get("tech") or pp.get("tech_stack") or "")):
                known.add(t.lower().strip())
    known.update(_COMMON_TOOLS_WHITELIST)
    return known


def _filter_skills_used(skills_csv: str, whitelist: set[str]) -> str:
    """Filter a comma-separated skills string to only include whitelisted skills."""
    items = _split_csv(skills_csv)
    filtered = [s for s in items if s.lower().strip() in whitelist]
    return ", ".join(_dedupe_keep_order(filtered))


def _coerce_skills(data: dict, profile: dict, jd_text: str) -> dict:
    raw_skills = data.get("skills") if isinstance(data.get("skills"), dict) else {}
    if not raw_skills and isinstance(data.get("technical_skills"), dict):
        raw_skills = data.get("technical_skills", {})

    # Map variant category labels into the schema expected by assembly/validator.
    mapped: dict[str, list[str]] = {
        "Languages": [],
        "Frameworks": [],
        "Developer Tools": [],
        "Databases and Libraries": [],
    }

    for key, value in raw_skills.items():
        key_l = str(key).lower()
        values = _split_csv(_csv_from_value(value))
        if "language" in key_l:
            mapped["Languages"].extend(values)
        elif "framework" in key_l:
            mapped["Frameworks"].extend(values)
        elif "database" in key_l or "librar" in key_l:
            mapped["Databases and Libraries"].extend(values)
        else:
            mapped["Developer Tools"].extend(values)

    # Build whitelist from boundary + preserved experience/projects
    boundary = profile.get("skills_boundary", {})
    all_known_lower = _build_skills_whitelist(profile)

    # Filter LLM-generated skills: only keep items in the whitelist
    for cat in mapped:
        mapped[cat] = [v for v in mapped[cat] if v.lower().strip() in all_known_lower]

    # Add boundary skills (guaranteed real)
    mapped["Languages"].extend([str(x) for x in boundary.get("languages", [])])
    mapped["Frameworks"].extend([str(x) for x in boundary.get("frameworks", [])])
    mapped["Developer Tools"].extend([str(x) for x in boundary.get("devops", [])])
    mapped["Developer Tools"].extend([str(x) for x in boundary.get("tools", [])])
    mapped["Databases and Libraries"].extend([str(x) for x in boundary.get("databases", [])])

    # Pull a capped set of JD keywords into tools for better ATS coverage.
    # Only add if they pass the whitelist check.
    existing_lower = set()
    for vals in mapped.values():
        existing_lower.update(v.lower().strip() for v in vals)
    try:
        jd_keywords = sorted(_extract_jd_keywords(jd_text or ""))
    except Exception:
        jd_keywords = []
    for k in jd_keywords:
        pk = _pretty_keyword(k)
        if _looks_like_technical_keyword(k) and pk.lower().strip() not in existing_lower:
            if pk.lower().strip() in all_known_lower:
                mapped["Developer Tools"].append(pk)
                existing_lower.add(pk.lower().strip())

    return {
        cat: ", ".join(_dedupe_keep_order(vals))
        for cat, vals in mapped.items()
    }


def _flatten_coursework(value) -> str:
    """Convert coursework to a clean comma-separated string (handles list or str)."""
    if isinstance(value, list):
        return ", ".join(sanitize_text(str(item)) for item in value if item)
    return str(value)


def _coerce_education(data: dict, profile: dict) -> dict:
    raw = data.get("education") if isinstance(data.get("education"), dict) else {}
    facts = profile.get("resume_facts", {})
    pres = facts.get("preserved_education", {})
    exp = profile.get("experience", {})

    return {
        "school": sanitize_text(str(pres.get("school") or raw.get("school") or facts.get("preserved_school") or "University")),
        "location": sanitize_text(str(pres.get("location") or raw.get("location") or "Toronto, ON")),
        "degree": sanitize_text(str(pres.get("degree") or raw.get("degree") or exp.get("education_level") or "Bachelor's Degree")),
        "dates": sanitize_text(str(pres.get("dates") or raw.get("dates") or "")),
        "coursework": sanitize_text(_flatten_coursework(raw.get("coursework") or pres.get("coursework") or "Software Engineering, Data Structures, Databases")),
    }


def _coerce_experience(data: dict, profile: dict, skill_hint: str) -> list[dict]:
    raw_entries = data.get("experience") if isinstance(data.get("experience"), list) else []
    preserved = profile.get("resume_facts", {}).get("preserved_companies", [])
    preserved_experience = profile.get("resume_facts", {}).get("preserved_experience", [])
    required_companies = [c for c in preserved if c and "code ninjas" not in c.lower()]

    if not required_companies:
        required_companies = [
            sanitize_text(str(e.get("company", ""))).strip()
            for e in raw_entries if isinstance(e, dict) and e.get("company")
        ]

    if not required_companies:
        required_companies = ["Previous Company"]

    # Build lookup from preserved_experience by company name (lowercase)
    preserved_lookup: dict[str, dict] = {}
    for pe in preserved_experience:
        if isinstance(pe, dict) and pe.get("company"):
            preserved_lookup[pe["company"].lower().strip()] = pe

    # Use LLM entries by index order — the LLM reorders by relevance but often
    # renames companies. We trust the LLM's BULLETS (they're tailored) but
    # enforce the real company names/titles from the profile.
    out: list[dict] = []
    whitelist = _build_skills_whitelist(profile)
    fallback_title = profile.get("experience", {}).get("target_role") or "Software Developer"
    for i, company in enumerate(required_companies[:4]):
        # Take LLM entry by index (LLM may rename companies but bullet content is tailored)
        src = raw_entries[i] if i < len(raw_entries) and isinstance(raw_entries[i], dict) else {}
        pres = preserved_lookup.get(company.lower().strip(), {})

        title = _preferred_title_for_company(company, sanitize_text(str(src.get("title") or fallback_title)))
        location = sanitize_text(str(pres.get("location") or src.get("location") or "Toronto, ON"))
        dates = sanitize_text(str(pres.get("dates") or src.get("dates") or ""))

        # Prefer LLM-rewritten bullets (tailored to JD), fallback to preserved originals
        llm_bullets = src.get("bullets") if isinstance(src.get("bullets"), list) and len(src.get("bullets", [])) >= 2 else None
        bullets = _normalize_bullets(llm_bullets or pres.get("bullets"), skill_hint)

        raw_skills_used = _csv_from_value(src.get("skills_used")) or _csv_from_value(pres.get("skills")) or skill_hint
        skills_used = _filter_skills_used(raw_skills_used, whitelist)
        if not skills_used:
            skills_used = skill_hint

        out.append({
            "company": company,
            "location": location,
            "title": title,
            "dates": dates,
            "bullets": bullets,
            "skills_used": skills_used,
        })

    return out


def _coerce_projects(data: dict, skill_hint: str, job_title: str, profile: dict | None = None) -> list[dict]:
    raw_projects = data.get("projects") if isinstance(data.get("projects"), list) else []
    preserved_projects = (profile or {}).get("resume_facts", {}).get("preserved_projects", [])
    whitelist = _build_skills_whitelist(profile or {})
    out: list[dict] = []

    # LLM fabricates projects to match the JD — names, tech stacks, and skills
    # can include ANY JD tool (candidate will build before interview).
    # Guardrails: reject placeholder names, cap tech_stack at 6 items.
    for i, project in enumerate(raw_projects[:3]):
        if not isinstance(project, dict):
            continue
        name = sanitize_text(str(project.get("name") or ""))
        # Reject placeholder-sounding names
        if not name or "project 1" in name.lower() or "project 2" in name.lower() or "targeted" in name.lower():
            name = ""
        tech_stack = _csv_from_value(project.get("tech_stack")) or ""
        skills_used_raw = _csv_from_value(project.get("skills_used")) or ""
        if not tech_stack and skills_used_raw:
            tech_stack = skills_used_raw
        # Cap tech_stack to 4 items to match original style (no whitelist filtering — projects use ANY JD tool)
        tech_items = _dedupe_keep_order(_split_csv(tech_stack))[:4]
        tech_stack = ", ".join(tech_items) if tech_items else ", ".join(_split_csv(skill_hint)[:4])
        dates = sanitize_text(str(project.get("dates") or "2024 -- Present"))
        bullets = _normalize_bullets(project.get("bullets"), skill_hint, min_count=3)[:3]
        # Projects allow ANY JD tool — no whitelist filtering on skills_used
        raw_su = skills_used_raw or tech_stack
        skills_used = ", ".join(_dedupe_keep_order(_split_csv(raw_su)))
        if not skills_used:
            skills_used = tech_stack
        if not name:
            name = _generate_plausible_project_name(tech_items, job_title, i)
        out.append({
            "name": name,
            "tech_stack": tech_stack,
            "dates": dates,
            "bullets": bullets,
            "skills_used": skills_used,
        })

    # Fallback: use preserved_projects if LLM didn't generate enough
    if len(out) < 2:
        existing_names_lower = {p["name"].lower().strip() for p in out}
        for pp in preserved_projects:
            if len(out) >= 2:
                break
            if not isinstance(pp, dict) or not pp.get("name"):
                continue
            if pp["name"].lower().strip() in existing_names_lower:
                continue
            fb_tech = _csv_from_value(pp.get("tech") or pp.get("tech_stack") or "")
            fb_skills = _csv_from_value(pp.get("skills") or pp.get("skills_used") or "")
            tech_items = _dedupe_keep_order(_split_csv(fb_tech or fb_skills))[:6]
            tech_str = ", ".join(tech_items) if tech_items else ", ".join(_split_csv(skill_hint)[:4])
            out.append({
                "name": pp["name"],
                "tech_stack": tech_str,
                "dates": pp.get("dates", "2024 -- Present"),
                "bullets": _normalize_bullets(pp.get("bullets"), skill_hint, min_count=2)[:3],
                "skills_used": _filter_skills_used(fb_skills or tech_str, whitelist) or tech_str,
            })

    return out[:3]


def _generate_plausible_project_name(tech_items: list[str], job_title: str, idx: int) -> str:
    """Generate a realistic-sounding project name from the tech stack and role."""
    # Map common tech to project types
    _PROJECT_TEMPLATES = [
        "Distributed Task Scheduler",
        "Real-Time Event Processing Pipeline",
        "Automated Testing Framework",
        "Full-Stack E-Commerce Platform",
        "Cloud Infrastructure Monitor",
        "API Gateway and Rate Limiter",
        "Machine Learning Model Server",
        "Container Orchestration Dashboard",
    ]
    return _PROJECT_TEMPLATES[idx % len(_PROJECT_TEMPLATES)]


def _coerce_tailor_schema(data: dict, profile: dict, job: dict) -> dict:
    """Normalize variant model output into the strict resume JSON schema."""
    jd_text = job.get("full_description") or ""
    title = sanitize_text(str(data.get("title") or job.get("title") or "Software Engineer"))

    skills = _coerce_skills(data, profile, jd_text)
    skill_pool = []
    for v in skills.values():
        skill_pool.extend(_split_csv(v))
    skill_hint = ", ".join(_dedupe_keep_order(skill_pool)[:10]) or "Python, SQL, Git, AWS"

    projects = _coerce_projects(data, skill_hint, title, profile=profile)

    # Inject project tech_stack items into the skills section so ATS finds them.
    # Projects can use ANY JD tool, and those must also appear in Technical Skills.
    proj_tools: list[str] = []
    for p in projects:
        proj_tools.extend(_split_csv(p.get("tech_stack", "")))
    existing_skills_lower = {s.lower().strip() for v in skills.values() for s in _split_csv(v)}
    for tool in _dedupe_keep_order(proj_tools):
        # Strip parenthetical fragments from items like "AWS (DynamoDB, ECS)" that got split by CSV parsing
        clean = re.sub(r'\s*\(.*$', '', tool).strip().rstrip(')')
        if not clean or len(clean) < 2:
            continue
        if clean.lower().strip() not in existing_skills_lower:
            skills["Developer Tools"] = (skills.get("Developer Tools", "") + ", " + clean).strip(", ")
            existing_skills_lower.add(clean.lower().strip())

    return {
        "title": title,
        "education": _coerce_education(data, profile),
        "experience": _coerce_experience(data, profile, skill_hint),
        "projects": projects,
        "skills": skills,
    }


def _ats_check_is_blocking() -> bool:
    """Whether ATS compliance failures should trigger retries before approval."""
    raw = os.environ.get("APPLYPILOT_TAILOR_ATS_BLOCKING", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _tailor_max_tokens() -> int:
    """Max output tokens for tailor JSON generation."""
    raw = os.environ.get("APPLYPILOT_TAILOR_MAX_TOKENS", "4096").strip()
    try:
        return max(1024, min(8192, int(raw)))
    except ValueError:
        return 4096


def _tailor_max_retries() -> int:
    """Per-job retry count for tailoring.  Minimum 1 retry (2 total iterations)
    to ensure every resume goes through at least one self-correction pass."""
    raw = os.environ.get("APPLYPILOT_TAILOR_MAX_RETRIES", "1").strip()
    try:
        return max(1, min(5, int(raw)))
    except ValueError:
        return 1


def _tailor_run_judge() -> bool:
    """Whether to run the extra LLM-judge pass after validation."""
    raw = os.environ.get("APPLYPILOT_TAILOR_RUN_JUDGE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _tailor_commit_every() -> int:
    """How many completed jobs to buffer before DB commit."""
    raw = os.environ.get("APPLYPILOT_TAILOR_COMMIT_EVERY", "1").strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return 1


# ── Resume Assembly (profile-driven header) ──────────────────────────────

def latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text content.

    Only use on user/LLM-generated text, NOT on LaTeX structure.
    """
    text = sanitize_text(str(text or ""))
    # Normalize problematic Unicode into pdflatex-safe ASCII equivalents.
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

    text = "".join(normalized_chars)
    text = re.sub(r"\s+", " ", text).strip()

    text = text.replace('\\', '\\textbackslash{}')
    for char, repl in [
        ('&', '\\&'), ('%', '\\%'), ('$', '\\$'), ('#', '\\#'),
        ('_', '\\_'), ('{', '\\{'), ('}', '\\}'),
        ('~', '\\textasciitilde{}'), ('^', '\\textasciicircum{}'),
    ]:
        text = text.replace(char, repl)
    return text


# ── Jake's Resume LaTeX Preamble ─────────────────────────────────────────

JAKE_PREAMBLE = r"""\documentclass[letterpaper,11pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\input{glyphtounicode}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-.5in}
\addtolength{\textheight}{1.0in}

\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\large
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]

\pdfgentounicode=1

%--- Custom commands ---
\newcommand{\resumeItem}[1]{
  \item\small{
    {#1 \vspace{-2pt}}
  }
}

\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubSubheading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \textit{\small#1} & \textit{\small #2} \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & #2 \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}

\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}"""


def assemble_resume_latex(data: dict, profile: dict) -> str:
    """Convert JSON resume data to a complete LaTeX document using Jake's Resume template.

    Header (name, contact) is ALWAYS code-injected from the profile,
    never LLM-generated. All text fields are LaTeX-escaped.

    Args:
        data: Parsed JSON resume from the LLM (new schema).
        profile: User profile dict from load_profile().

    Returns:
        Complete LaTeX document string (.tex content).
    """
    personal = profile.get("personal", {})
    parts: list[str] = [JAKE_PREAMBLE, "", "\\begin{document}", ""]

    # ── Header (code-injected from profile) ──
    name = latex_escape(personal.get("full_name", ""))
    phone = personal.get("phone", "")
    email = personal.get("email", "")
    linkedin = personal.get("linkedin_url", "")
    github = personal.get("github_url", "")

    parts.append("\\begin{center}")
    parts.append(f"    \\textbf{{\\Huge \\scshape {name}}} \\\\ \\vspace{{1pt}}")

    contact_items: list[str] = []
    if phone:
        contact_items.append(f"\\small {latex_escape(phone)}")
    if email:
        contact_items.append(
            f"\\href{{mailto:{email}}}{{\\underline{{{latex_escape(email)}}}}}"
        )
    if linkedin:
        display = linkedin.replace("https://", "").replace("http://", "")
        contact_items.append(f"\\href{{{linkedin}}}{{\\underline{{{latex_escape(display)}}}}}")
    if github:
        display = github.replace("https://", "").replace("http://", "")
        contact_items.append(f"\\href{{{github}}}{{\\underline{{{latex_escape(display)}}}}}")

    if contact_items:
        parts.append(f"    {' $|$ '.join(contact_items)}")
    parts.append("\\end{center}")
    parts.append("")

    # ── Education ──
    edu = data.get("education", {})
    if isinstance(edu, dict) and edu:
        parts.append("%-----------EDUCATION-----------")
        parts.append("\\section{Education}")
        parts.append("  \\resumeSubHeadingListStart")
        parts.append("    \\resumeSubheading")
        parts.append(
            f"      {{{latex_escape(edu.get('school', ''))}}}"
            f"{{{latex_escape(edu.get('location', ''))}}}"
        )
        parts.append(
            f"      {{{latex_escape(edu.get('degree', ''))}}}"
            f"{{{latex_escape(edu.get('dates', ''))}}}"
        )
        coursework = edu.get("coursework", "")
        if coursework:
            parts.append("      \\resumeItemListStart")
            parts.append(
                f"        \\resumeItem{{\\textbf{{Coursework}}: {latex_escape(coursework)}}}"
            )
            parts.append("      \\resumeItemListEnd")
        parts.append("  \\resumeSubHeadingListEnd")
        parts.append("")

    # ── Experience ──
    experience = data.get("experience", [])
    if experience:
        parts.append("%-----------EXPERIENCE-----------")
        parts.append("\\section{Experience}")
        parts.append("  \\resumeSubHeadingListStart")
        for entry in experience:
            parts.append("    \\resumeSubheading")
            parts.append(
                f"      {{{latex_escape(entry.get('company', ''))}}}"
                f"{{{latex_escape(entry.get('location', ''))}}}"
            )
            parts.append(
                f"      {{{latex_escape(entry.get('title', ''))}}}"
                f"{{{latex_escape(entry.get('dates', ''))}}}"
            )
            parts.append("      \\resumeItemListStart")
            for bullet in entry.get("bullets", []):
                parts.append(
                    f"        \\resumeItem{{{latex_escape(sanitize_text(bullet))}}}"
                )
            # Skills used -- last line under each experience entry
            skills_used = entry.get("skills_used", "")
            if skills_used:
                parts.append(
                    f"        \\resumeItem{{\\textbf{{Skills}}: {latex_escape(skills_used)}}}"
                )
            parts.append("      \\resumeItemListEnd")
            parts.append("")
        parts.append("  \\resumeSubHeadingListEnd")
        parts.append("")

    # ── Projects ──
    projects = data.get("projects", [])
    if projects:
        parts.append("%-----------PROJECTS-----------")
        parts.append("\\section{Projects}")
        parts.append("  \\resumeSubHeadingListStart")
        for entry in projects:
            pname = latex_escape(entry.get("name", ""))
            tech = latex_escape(entry.get("tech_stack", ""))
            dates = latex_escape(entry.get("dates", ""))
            parts.append("    \\resumeProjectHeading")
            parts.append(
                f"      {{\\textbf{{{pname}}} $|$ \\emph{{\\small {tech}}}}}{{{dates}}}"
            )
            parts.append("      \\resumeItemListStart")
            for bullet in entry.get("bullets", []):
                parts.append(
                    f"        \\resumeItem{{{latex_escape(sanitize_text(bullet))}}}"
                )
            # Skills bullet in projects
            skills_used = entry.get("skills_used", "")
            if skills_used:
                parts.append(
                    f"        \\resumeItem{{\\textbf{{Skills}}: {latex_escape(skills_used)}}}"
                )
            parts.append("      \\resumeItemListEnd")
            parts.append("")
        parts.append("  \\resumeSubHeadingListEnd")
        parts.append("")

    # ── Technical Skills ──
    skills = data.get("skills", {})
    if isinstance(skills, dict) and skills:
        parts.append("%-----------TECHNICAL SKILLS-----------")
        parts.append("\\section{Technical Skills}")
        parts.append(" \\begin{itemize}[leftmargin=0.15in, label={}]")
        parts.append("    \\small{\\item{")
        skill_entries: list[str] = []
        for cat, val in skills.items():
            skill_entries.append(
                f"     \\textbf{{{latex_escape(cat)}}}{{: {latex_escape(str(val))}}}"
            )
        parts.append(" \\\\\n".join(skill_entries))
        parts.append("    }}")
        parts.append(" \\end{itemize}")
        parts.append("")

    parts.append("\\end{document}")
    return "\n".join(parts)


def assemble_resume_text(data: dict, profile: dict) -> str:
    """Convert JSON resume data to formatted plain text (for judge comparison).

    Header (name, contact) is ALWAYS code-injected from the profile,
    never LLM-generated. All text fields are sanitized.

    Args:
        data: Parsed JSON resume from the LLM (new schema).
        profile: User profile dict from load_profile().

    Returns:
        Formatted resume text.
    """
    personal = profile.get("personal", {})
    lines: list[str] = []

    # Header -- always code-injected from profile
    lines.append(personal.get("full_name", ""))
    lines.append(sanitize_text(data.get("title", "Software Engineer")))

    # Contact line
    contact_parts: list[str] = []
    if personal.get("email"):
        contact_parts.append(personal["email"])
    if personal.get("phone"):
        contact_parts.append(personal["phone"])
    if personal.get("github_url"):
        contact_parts.append(personal["github_url"])
    if personal.get("linkedin_url"):
        contact_parts.append(personal["linkedin_url"])
    if contact_parts:
        lines.append(" | ".join(contact_parts))
    lines.append("")

    # Education
    lines.append("EDUCATION")
    edu = data.get("education", {})
    if isinstance(edu, dict):
        lines.append(f"{edu.get('school', '')} | {edu.get('degree', '')} | {edu.get('dates', '')}")
        if edu.get("coursework"):
            lines.append(f"Coursework: {edu['coursework']}")
    else:
        lines.append(sanitize_text(str(edu)))
    lines.append("")

    # Experience
    lines.append("EXPERIENCE")
    for entry in data.get("experience", []):
        lines.append(f"{entry.get('title', '')} at {entry.get('company', '')}")
        lines.append(f"{entry.get('location', '')} | {entry.get('dates', '')}")
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        if entry.get("skills_used"):
            lines.append(f"Skills: {entry['skills_used']}")
        lines.append("")

    # Projects
    lines.append("PROJECTS")
    for entry in data.get("projects", []):
        lines.append(f"{entry.get('name', '')} | {entry.get('tech_stack', '')}")
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        if entry.get("skills_used"):
            lines.append(f"Skills: {entry['skills_used']}")
        lines.append("")

    # Technical Skills
    lines.append("TECHNICAL SKILLS")
    if isinstance(data.get("skills"), dict):
        for cat, val in data["skills"].items():
            lines.append(f"{cat}: {sanitize_text(str(val))}")

    return "\n".join(lines)


# ── LLM Judge ────────────────────────────────────────────────────────────

def judge_tailored_resume(
    original_text: str, tailored_text: str, job_title: str, profile: dict
) -> dict:
    """LLM judge layer: catches subtle fabrication that programmatic checks miss.

    Args:
        original_text: Base resume text.
        tailored_text: Tailored resume text.
        job_title: Target job title.
        profile: User profile for building the judge prompt.

    Returns:
        {"passed": bool, "verdict": str, "issues": str, "raw": str}
    """
    judge_prompt = _build_judge_prompt(profile)

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": (
            f"JOB TITLE: {job_title}\n\n"
            f"ORIGINAL RESUME:\n{original_text}\n\n---\n\n"
            f"TAILORED RESUME:\n{tailored_text}\n\n"
            "Judge this tailored resume:"
        )},
    ]

    client = get_client()
    response = client.chat(messages, max_tokens=2048, temperature=0.1)

    passed = "VERDICT: PASS" in response.upper()
    issues = "none"
    if "ISSUES:" in response.upper():
        issues_idx = response.upper().index("ISSUES:")
        issues = response[issues_idx + 7:].strip()

    return {
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "issues": issues,
        "raw": response,
    }


# ── Core Tailoring ───────────────────────────────────────────────────────

def tailor_resume(
    resume_text: str,
    job: dict,
    profile: dict,
    max_retries: int = 3,
    validation_mode: str = "normal",
) -> tuple[str, str, dict]:
    """Generate a tailored resume via JSON output + fresh context on each retry.

    Key design choices:
    - LLM returns structured JSON, code assembles LaTeX (Jake's Resume template)
    - Plain text is also assembled for the LLM judge layer
    - Each retry starts a FRESH conversation (no apologetic spiral)
    - Issues from previous attempts are noted in the system prompt
    - Em dashes and smart quotes are auto-fixed, not rejected

    Args:
        resume_text: Base resume text.
        job: Job dict with title, site, location, full_description.
        profile: User profile dict.
        max_retries: Maximum retry attempts.

    Returns:
        (tailored_latex, tailored_text, report) where tailored_latex is the .tex
        content, tailored_text is plain text for judge, and report has validation details.
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    report: dict = {
        "attempts": 0,
        "validator": None,
        "ats_check": None,
        "ats_rendered_check": None,
        "judge": None,
        "status": "pending",
    }
    avoid_notes: list[str] = []
    tailored_latex = ""
    tailored_text = ""
    client = get_client()
    tailor_prompt_base = _build_tailor_prompt(profile)
    ats_blocking = _ats_check_is_blocking()

    for attempt in range(max_retries + 1):
        report["attempts"] = attempt + 1

        # Fresh conversation every attempt
        prompt = tailor_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES (from previous attempt):\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"ORIGINAL RESUME:\n{resume_text}\n\n---\n\nTARGET JOB:\n{job_text}\n\nReturn the JSON:"},
        ]

        raw = client.chat(messages, max_tokens=_tailor_max_tokens(), temperature=0.0)

        # Parse JSON from response
        try:
            data = extract_json(raw)
        except ValueError:
            avoid_notes.append("Output was not valid JSON. Return ONLY a JSON object, nothing else.")
            continue

        # Normalize provider-specific variants into the strict schema we validate/render.
        data = _coerce_tailor_schema(data, profile, job)

        # Layer 1: Validate JSON fields
        validation = validate_json_fields(
            data,
            profile,
            jd_text=job.get("full_description", ""),
            validation_mode=validation_mode,
        )
        report["validator"] = validation

        if not validation["passed"]:
            avoid_notes.extend(validation["errors"])
            if attempt < max_retries:
                continue
            # Last attempt -- assemble whatever we got
            tailored_latex = assemble_resume_latex(data, profile)
            tailored_text = assemble_resume_text(data, profile)
            report["status"] = "failed_validation"
            return tailored_latex, tailored_text, report

        # Layer 2: ATS compliance check (keyword coverage + bullet formula)
        # This is a BLOCKING check -- failures trigger a retry with specific feedback
        jd_full = job.get("full_description", "")
        ats_check = validate_ats_compliance(data, jd_full)
        report["ats_check"] = ats_check

        if not ats_check["passed"]:
            avoid_notes.extend(ats_check["errors"])
            log.debug(
                "ATS compliance FAIL for %s (attempt %d): kw=%.0f%% bullets=%.0f%% | %s",
                job.get("title", "")[:40], attempt + 1,
                ats_check["keyword_coverage"] * 100,
                ats_check["bullet_compliance"] * 100,
                "; ".join(ats_check["errors"])[:200],
            )
            if ats_blocking:
                if attempt < max_retries:
                    continue
                # Last attempt -- accept what we have but log the gap
                log.warning(
                    "ATS check failed after all retries for %s: kw=%.0f%% bullets=%.0f%%",
                    job.get("title", "")[:40],
                    ats_check["keyword_coverage"] * 100,
                    ats_check["bullet_compliance"] * 100,
                )
                # Keep artifacts for inspection, but do not approve this resume.
                tailored_latex = assemble_resume_latex(data, profile)
                tailored_text = assemble_resume_text(data, profile)
                report["status"] = "failed_validation"
                return tailored_latex, tailored_text, report
            else:
                log.info(
                    "ATS check advisory-only for %s: kw=%.0f%% bullets=%.0f%%",
                    job.get("title", "")[:40],
                    ats_check["keyword_coverage"] * 100,
                    ats_check["bullet_compliance"] * 100,
                )

        # Assemble both LaTeX and plain text
        tailored_latex = assemble_resume_latex(data, profile)
        tailored_text = assemble_resume_text(data, profile)

        # Layer 2a-post: Quality gate on assembled LaTeX (catches placeholders, generic content)
        tex_quality = validate_tex_quality(tailored_latex)
        report["tex_quality"] = tex_quality
        if not tex_quality["passed"]:
            log.warning(
                "Tex quality gate FAIL for %s (attempt %d): %s",
                job.get("title", "")[:40], attempt + 1,
                "; ".join(tex_quality["issues"])[:300],
            )
            avoid_notes.extend(tex_quality["issues"])
            if attempt < max_retries:
                continue

        # Layer 2b: Post-assembly ATS check on rendered text.
        # This re-validates STEP 3 style constraints after JSON -> text/LaTeX
        # conversion so we can iterate again when formatting drift appears.
        ats_rendered_check = validate_ats_rendered_text(tailored_text, jd_full)
        report["ats_rendered_check"] = ats_rendered_check
        if not ats_rendered_check["passed"]:
            avoid_notes.extend(ats_rendered_check["errors"])
            log.debug(
                "Rendered ATS FAIL for %s (attempt %d): kw=%.0f%% bullets=%.0f%% | %s",
                job.get("title", "")[:40], attempt + 1,
                ats_rendered_check["keyword_coverage"] * 100,
                ats_rendered_check["bullet_compliance"] * 100,
                "; ".join(ats_rendered_check["errors"])[:200],
            )
            if ats_blocking:
                if attempt < max_retries:
                    continue
                log.warning(
                    "Rendered ATS check failed after all retries for %s: kw=%.0f%% bullets=%.0f%%",
                    job.get("title", "")[:40],
                    ats_rendered_check["keyword_coverage"] * 100,
                    ats_rendered_check["bullet_compliance"] * 100,
                )
                report["status"] = "failed_validation"
                return tailored_latex, tailored_text, report

        # Layer 3: Optional LLM judge (advisory only).
        if _tailor_run_judge():
            try:
                judge = judge_tailored_resume(resume_text, tailored_text, job.get("title", ""), profile)
                report["judge"] = judge
                if not judge["passed"]:
                    log.debug("Judge advisory FAIL for %s: %s", job.get("title", "")[:40], judge["issues"][:200])
            except Exception as e:
                report["judge"] = {"passed": False, "verdict": "ERROR", "issues": str(e), "raw": ""}
                log.debug("Judge call failed for %s: %s", job.get("title", "")[:40], e)
        else:
            report["judge"] = {"passed": True, "verdict": "SKIPPED", "issues": "judge disabled", "raw": ""}

        # Validation + ATS check passed → approve (but enforce 2nd iteration when bullets are weak)
        if attempt == 0 and max_retries >= 1:
            kw_pct = ats_check.get("keyword_coverage", 0) * 100
            bl_pct = ats_check.get("bullet_compliance", 0) * 100

            # Only force 2nd iteration if bullet compliance is below 75%.
            # When 1st pass is already decent, the 2nd iteration tends to
            # degrade bullet structure while chasing keyword density.
            if bl_pct < 75:
                bad_bullet_list = ats_check.get("bad_bullets", [])
                bad_bullet_detail = ""
                if bad_bullet_list:
                    bad_bullet_detail = " Specific bad bullets: " + "; ".join(b[:100] for b in bad_bullet_list[:5])
                avoid_notes.append(
                    f"First pass: kw={kw_pct:.0f}% bullets={bl_pct:.0f}%. "
                    "PRIORITY 1 — BULLET FORMULA: Re-read EVERY bullet in experience AND projects. "
                    "Each MUST end with 'resulting in [measurable impact]'. "
                    "Do NOT remove any keywords or skills to make room. "
                    "Do NOT rewrite bullets that already follow the formula. "
                    "ONLY fix bullets that are missing 'resulting in'."
                    f"{bad_bullet_detail}"
                )
                log.info(
                    "Mandatory 2nd iteration for %s (1st pass kw=%.0f%% bullets=%.0f%%)",
                    job.get("title", "")[:40], kw_pct, bl_pct,
                )
                continue
            else:
                log.info(
                    "Skipping 2nd iteration for %s (1st pass kw=%.0f%% bullets=%.0f%% — above threshold)",
                    job.get("title", "")[:40], kw_pct, bl_pct,
                )

        report["status"] = "approved"
        return tailored_latex, tailored_text, report

    report["status"] = "exhausted_retries"
    return tailored_latex, tailored_text, report


# ── Batch Entry Point ────────────────────────────────────────────────────

def _make_file_prefix(job: dict) -> str:
    """Build a safe, unique filename prefix for a job's output files."""
    safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
    safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
    id_source = str(job.get("url") or job.get("application_url") or f"{job.get('site','')}-{job.get('title','')}")
    suffix = hashlib.sha1(id_source.encode("utf-8")).hexdigest()[:10]
    return f"{safe_site}_{safe_title}_{suffix}"


def _try_rezzy_tailor(job: dict, prefix: str) -> dict | None:
    """Attempt to tailor via Rezzy API. Returns result dict or None to fallback."""
    from applypilot.scoring.rezzy import is_rezzy_enabled, rezzy_tailor_resume, RezzyError

    if not is_rezzy_enabled():
        return None

    try:
        result = rezzy_tailor_resume(job, TAILORED_DIR, prefix)
    except RezzyError as e:
        log.warning("Rezzy error for %s — falling back to Copilot: %s", job["title"][:40], e)
        return None

    if result is None:
        # Credits exhausted — disable Rezzy for the rest of this process
        os.environ["APPLYPILOT_USE_REZZY"] = "0"
        log.warning("Rezzy credits exhausted — disabling Rezzy for remaining jobs")
        return None

    log.info("[REZZY] %s — PDF ready: %s", job["title"][:40], result["pdf_path"])
    return {
        "url": job["url"],
        "path": result["tex_path"],
        "pdf_path": result["pdf_path"],
        "title": job["title"],
        "site": job["site"],
        "status": "approved",
        "attempts": 1,
        "source": "rezzy",
    }


def _process_one_tailor(
    job: dict,
    resume_text: str,
    profile: dict,
    validation_mode: str = "normal",
) -> dict:
    """Process a single job for tailoring (thread-safe).

    Tries Rezzy API first (produces its own PDF). Falls back to
    Copilot LLM pipeline if Rezzy is unavailable or credits exhausted.
    """
    try:
        prefix = _make_file_prefix(job)

        # --- Rezzy (primary) ---
        rezzy_result = _try_rezzy_tailor(job, prefix)
        if rezzy_result is not None:
            return rezzy_result

        # --- Copilot LLM (fallback) ---
        tailored_latex, tailored_text, report = tailor_resume(
            resume_text,
            job,
            profile,
            max_retries=_tailor_max_retries(),
            validation_mode=validation_mode,
        )

        tex_path = TAILORED_DIR / f"{prefix}.tex"
        tex_path.write_text(tailored_latex, encoding="utf-8")

        txt_path = TAILORED_DIR / f"{prefix}.txt"
        txt_path.write_text(tailored_text, encoding="utf-8")

        job_path = TAILORED_DIR / f"{prefix}_JOB.txt"
        job_desc = (
            f"Title: {job['title']}\n"
            f"Company: {job['site']}\n"
            f"Location: {job.get('location', 'N/A')}\n"
            f"Score: {job.get('fit_score', 'N/A')}\n"
            f"URL: {job['url']}\n\n"
            f"{job.get('full_description', '')}"
        )
        job_path.write_text(job_desc, encoding="utf-8")

        report_path = TAILORED_DIR / f"{prefix}_REPORT.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        pdf_path = None
        if report["status"] == "approved":
            try:
                from applypilot.scoring.pdf import compile_latex_to_pdf
                pdf_path = str(compile_latex_to_pdf(tex_path))
            except Exception:
                log.debug("PDF generation failed for %s", tex_path, exc_info=True)

        return {
            "url": job["url"],
            "path": str(tex_path),
            "pdf_path": pdf_path,
            "title": job["title"],
            "site": job["site"],
            "status": report["status"],
            "attempts": report["attempts"],
            "source": "copilot",
        }
    except Exception as e:
        log.error("[ERROR] %s -- %s", job["title"][:40], e)
        return {
            "url": job["url"], "title": job["title"], "site": job["site"],
            "status": "error", "attempts": 0, "path": None, "pdf_path": None,
        }


def run_tailoring(
    min_score: int = 7,
    limit: int = 20,
    workers: int = 1,
    validation_mode: str = "normal",
) -> dict:
    """Generate tailored resumes for high-scoring jobs.

    Args:
        min_score: Minimum fit_score to tailor for.
        limit: Maximum jobs to process.
        workers: Number of parallel workers.
        validation_mode: Validation strictness passed into JSON/text validators.

    Returns:
        {"approved": int, "failed": int, "errors": int, "elapsed": float}
    """
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=min_score, limit=limit)

    if not jobs:
        log.info("No untailored jobs with score >= %d.", min_score)
        return {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0}

    TAILORED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Tailoring resumes for %d jobs (score >= %d, %d workers)...", len(jobs), min_score, workers)
    t0 = time.time()
    completed = 0
    _pending_commit: list[dict] = []
    COMMIT_EVERY = _tailor_commit_every()
    results: list[dict] = []
    stats: dict[str, int] = {"approved": 0, "failed_validation": 0, "failed_judge": 0, "error": 0}
    _lock = threading.Lock()

    def _flush_pending(pending: list[dict]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for r in pending:
            if r["status"] == "approved":
                conn.execute(
                    "UPDATE jobs SET tailored_resume_path=?, tailored_at=?, "
                    "tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                    (r["path"], now, r["url"]),
                )
            elif r["status"] == "error":
                # Transient LLM errors: do NOT increment attempt counter
                # so the job stays eligible for retry on next run.
                pass
            else:
                # Permanent failures (failed_validation, failed_judge):
                # increment attempt counter toward exhaustion.
                conn.execute(
                    "UPDATE jobs SET tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                    (r["url"],),
                )
        conn.commit()

    def _on_result(result: dict):
        nonlocal completed
        with _lock:
            completed += 1
            results.append(result)
            _pending_commit.append(result)
            stats[result.get("status", "error")] = stats.get(result.get("status", "error"), 0) + 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log.info(
                "%d/%d [%s] attempts=%s | %.1f jobs/min | %s",
                completed, len(jobs),
                result["status"].upper(),
                result.get("attempts", "?"),
                rate * 60,
                result["title"][:40],
            )
            if len(_pending_commit) >= COMMIT_EVERY:
                _flush_pending(_pending_commit.copy())
                _pending_commit.clear()

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_one_tailor,
                    job,
                    resume_text,
                    profile,
                    validation_mode,
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                _on_result(future.result())
    else:
        for job in jobs:
            _on_result(_process_one_tailor(job, resume_text, profile, validation_mode))

    # Flush any remaining uncommitted results
    if _pending_commit:
        _flush_pending(_pending_commit)

    elapsed = time.time() - t0
    log.info(
        "Tailoring done in %.1fs: %d approved, %d failed_validation, %d failed_judge, %d errors",
        elapsed,
        stats.get("approved", 0),
        stats.get("failed_validation", 0),
        stats.get("failed_judge", 0),
        stats.get("error", 0),
    )

    return {
        "approved": stats.get("approved", 0),
        "failed": stats.get("failed_validation", 0) + stats.get("failed_judge", 0),
        "errors": stats.get("error", 0),
        "elapsed": elapsed,
    }
