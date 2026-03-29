"""Prompt builder for the autonomous job application agent.

Constructs the full instruction prompt that tells Claude Code / the AI agent
how to fill out a job application form using Playwright MCP tools. All
personal data is loaded from the user's profile -- nothing is hardcoded.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from applypilot import config

logger = logging.getLogger(__name__)


def _build_profile_summary(profile: dict) -> str:
    """Format the applicant profile section of the prompt.

    Reads all relevant fields from the profile dict and returns a
    human-readable multi-line summary for the agent.
    """
    p = profile
    personal = p["personal"]
    work_auth = p["work_authorization"]
    comp = p["compensation"]
    exp = p.get("experience", {})
    avail = p.get("availability", {})
    eeo = p.get("eeo_voluntary", {})

    lines = [
        f"Name: {personal['full_name']}",
        f"Email: {personal['email']}",
        f"Phone: {personal['phone']}",
    ]

    # Address -- handle optional fields gracefully
    addr_parts = [
        personal.get("address", ""),
        personal.get("city", ""),
        personal.get("province_state", ""),
        personal.get("country", ""),
        personal.get("postal_code", ""),
    ]
    lines.append(f"Address: {', '.join(p for p in addr_parts if p)}")

    if personal.get("linkedin_url"):
        lines.append(f"LinkedIn: {personal['linkedin_url']}")
    if personal.get("github_url"):
        lines.append(f"GitHub: {personal['github_url']}")
    if personal.get("portfolio_url"):
        lines.append(f"Portfolio: {personal['portfolio_url']}")
    if personal.get("website_url"):
        lines.append(f"Website: {personal['website_url']}")

    # Work authorization
    lines.append(f"Work Auth: {work_auth.get('legally_authorized_to_work', 'See profile')}")
    lines.append(f"Sponsorship Needed: {work_auth.get('require_sponsorship', 'See profile')}")
    if work_auth.get("work_permit_type"):
        lines.append(f"Work Permit: {work_auth['work_permit_type']}")

    # Compensation
    currency = comp.get("salary_currency", "USD")
    lines.append(f"Salary Expectation: ${comp['salary_expectation']} {currency}")

    # Experience
    if exp.get("years_of_experience_total"):
        lines.append(f"Years Experience: {exp['years_of_experience_total']}")
    if exp.get("education_level"):
        lines.append(f"Education: {exp['education_level']}")

    # Availability
    lines.append(f"Available: {avail.get('earliest_start_date', 'Immediately')}")

    # Standard responses
    lines.extend([
        "Age 18+: Yes",
        "Background Check: Yes",
        "Felony: No",
        "Previously Worked Here: No",
        "How Heard: Online Job Board",
    ])

    # EEO
    lines.append(f"Gender: {eeo.get('gender', 'Decline to self-identify')}")
    lines.append(f"Race: {eeo.get('race_ethnicity', 'Decline to self-identify')}")
    lines.append(f"Veteran: {eeo.get('veteran_status', 'I am not a protected veteran')}")
    lines.append(f"Disability: {eeo.get('disability_status', 'I do not wish to answer')}")

    return "\n".join(lines)


def _build_location_check(profile: dict, search_config: dict) -> str:
    """Build the location eligibility check section of the prompt.

    Uses the accept_patterns from search config to determine which cities
    are acceptable for hybrid/onsite roles.
    """
    personal = profile["personal"]
    location_cfg = search_config.get("location", {})
    accept_patterns = location_cfg.get("accept_patterns", [])
    primary_city = personal.get("city", location_cfg.get("primary", "your city"))

    # Build the list of acceptable cities for hybrid/onsite
    if accept_patterns:
        city_list = ", ".join(accept_patterns)
    else:
        city_list = primary_city

    return f"""== LOCATION ==
Only apply to jobs located in CANADA or marked as Remote. If the job is clearly located in the United States or another country (not Canada) and is NOT remote, output RESULT:FAILED:not_eligible_location.
If asked about relocation or location in screening questions, answer that {primary_city} is the current location. If asked "are you willing to relocate?", answer Yes but only within Canada. If a location dropdown is required, pick the closest Canadian option.
Acceptable locations: Any Canadian city/province, Remote, Anywhere, Work from home, Hybrid."""


def _build_salary_section(profile: dict) -> str:
    """Build the salary negotiation instructions.

    Adapts floor, range, and currency from the profile's compensation section.
    """
    comp = profile["compensation"]
    currency = comp.get("salary_currency", "USD")
    floor = comp["salary_expectation"]
    range_min = comp.get("salary_range_min", floor)
    range_max = comp.get("salary_range_max", str(int(floor) + 20000) if floor.isdigit() else floor)
    conversion_note = comp.get("currency_conversion_note", "")

    # Compute example hourly rates at 3 salary levels
    try:
        floor_int = int(floor)
        examples = [
            (f"${floor_int // 1000}K", floor_int // 2080),
            (f"${(floor_int + 25000) // 1000}K", (floor_int + 25000) // 2080),
            (f"${(floor_int + 55000) // 1000}K", (floor_int + 55000) // 2080),
        ]
        hourly_line = ", ".join(f"{sal} = ${hr}/hr" for sal, hr in examples)
    except (ValueError, TypeError):
        hourly_line = "Divide annual salary by 2080"

    # Currency conversion guidance
    if conversion_note:
        convert_line = f"Posting is in a different currency? -> {conversion_note}"
    else:
        convert_line = "Posting is in a different currency? -> Target midpoint of their range. Convert if needed."

    return f"""== SALARY (think, don't just copy) ==
${floor} {currency} is the FLOOR. Never go below it. But don't always use it either.

Decision tree:
1. Job posting shows a range (e.g. "$120K-$160K")? -> Answer with the MIDPOINT ($140K).
2. Title says Senior, Staff, Lead, Principal, Architect, or level II/III/IV? -> Minimum $110K {currency}. Use midpoint of posted range if higher.
3. {convert_line}
4. No salary info anywhere? -> Use ${floor} {currency}.
5. Asked for a range? -> Give posted midpoint minus 10% to midpoint plus 10%. No posted range? -> "${range_min}-${range_max} {currency}".
6. Hourly rate? -> Divide your annual answer by 2080. ({hourly_line})"""


def _build_screening_section(profile: dict) -> str:
    """Build the screening questions guidance section."""
    personal = profile["personal"]
    exp = profile.get("experience", {})
    city = personal.get("city", "their city")
    years = exp.get("years_of_experience_total", "multiple")
    target_role = exp.get("target_role", personal.get("current_job_title", "software engineer"))
    work_auth = profile["work_authorization"]

    return f"""== SCREENING QUESTIONS (be strategic) ==
Hard facts -> answer truthfully from the profile. No guessing. This includes:
  - Location/relocation: lives in {city}, cannot relocate
  - Work authorization: {work_auth.get('legally_authorized_to_work', 'see profile')}
  - Citizenship, clearance, licenses, certifications: answer from profile only
  - Criminal/background: answer from profile only

Skills and tools -> be confident. I am a {target_role} with {years} years experience. If the question asks "Do you have experience with [tool]?" and it's in the same domain (DevOps, backend, ML, cloud, automation), answer YES. Software engineers learn tools fast. Don't sell short.

Open-ended questions ("Why do you want this role?", "Tell us about yourself", "What interests you?") -> Write 2-3 sentences in first person as me. Be specific to THIS job. Reference something from the job description. Connect it to a real achievement from my resume. Sound natural and human. No corporate buzzwords. No "I am passionate about..." -- write like I would in a casual-but-professional tone. Examples of good tone: "I've spent the last few years building distributed systems and this role's focus on scalability caught my eye" or "Your team's work on X aligns with what I've been doing at Y".

EEO/demographics -> "Decline to self-identify" or "Prefer not to say" for everything."""


def _build_hard_rules(profile: dict) -> str:
    """Build the hard rules section with work auth and name from profile."""
    personal = profile["personal"]
    work_auth = profile["work_authorization"]

    full_name = personal["full_name"]
    preferred_name = personal.get("preferred_name", full_name.split()[0])
    preferred_last = full_name.split()[-1] if " " in full_name else ""
    display_name = f"{preferred_name} {preferred_last}".strip() if preferred_last else preferred_name

    # Build work auth rule dynamically
    auth_info = work_auth.get("legally_authorized_to_work", "")
    sponsorship = work_auth.get("require_sponsorship", "")
    permit_type = work_auth.get("work_permit_type", "")

    work_auth_rule = "Work auth: Answer truthfully from profile."
    if permit_type:
        work_auth_rule = f"Work auth: {permit_type}. Sponsorship needed: {sponsorship}."

    name_rule = f'Name: Legal name = {full_name}.'
    if preferred_name and preferred_name != full_name.split()[0]:
        name_rule += f' Preferred name = {preferred_name}. Use "{display_name}" unless a field specifically says "legal name".'

    return f"""== HARD RULES (never break these) ==
1. Never lie about: citizenship, work authorization, criminal history, education credentials, security clearance, licenses.
2. {work_auth_rule}
3. {name_rule}"""


def _build_captcha_section() -> str:
    """Build the CAPTCHA detection and solving instructions.

    Reads the CapSolver API key from environment. The CAPTCHA section
    contains no personal data -- it's the same for every user.
    """
    config.load_env()
    capsolver_key = os.environ.get("CAPSOLVER_API_KEY", "")

    return f"""== CAPTCHA ==
You solve CAPTCHAs via the CapSolver REST API. No browser extension. You control the entire flow.
API key: {capsolver_key or 'NOT CONFIGURED — skip to MANUAL FALLBACK for all CAPTCHAs'}
API base: https://api.capsolver.com

ABSOLUTE RULES -- NEVER VIOLATE THESE:
- NEVER click a reCAPTCHA checkbox, hCaptcha checkbox, or any CAPTCHA widget/iframe. Clicking triggers a visual image challenge that is UNSOLVABLE. The CapSolver API solves CAPTCHAs server-side without any visual interaction.
- NEVER try to solve CAPTCHAs visually (no clicking images, no audio challenges, no puzzles). You cannot see images or hear audio.
- NEVER say "CapSolver is not configured" -- the API key is embedded above. If you see a key starting with CAP-, it IS configured.

When ANY CAPTCHA appears (hCaptcha, reCAPTCHA, Turnstile -- regardless of what it looks like visually), you MUST:
1. Run CAPTCHA DETECT (browser_evaluate) to get the type and sitekey
2. Run CAPTCHA SOLVE (createTask -> poll -> inject) with the CapSolver API
3. ONLY go to MANUAL FALLBACK if CapSolver createTask returns errorId > 0
Do NOT skip the API call based on what the CAPTCHA looks like. CapSolver solves CAPTCHAs server-side -- it does NOT need to see or interact with images, puzzles, or games. Even "drag the pipe" or "click all traffic lights" hCaptchas are solved via API token, not visually. ALWAYS try the API first.

--- CAPTCHA DETECT ---
Run this browser_evaluate after every navigation, Apply/Submit/Login click, or when a page feels stuck.
IMPORTANT: Detection order matters. hCaptcha elements also have data-sitekey, so check hCaptcha BEFORE reCAPTCHA.

browser_evaluate function: () => {{{{
  const r = {{}};
  const url = window.location.href;
  // 1. hCaptcha (check FIRST -- hCaptcha uses data-sitekey too)
  const hc = document.querySelector('.h-captcha, [data-hcaptcha-sitekey]');
  if (hc) {{{{
    r.type = 'hcaptcha'; r.sitekey = hc.dataset.sitekey || hc.dataset.hcaptchaSitekey;
  }}}}
  if (!r.type && document.querySelector('script[src*="hcaptcha.com"], iframe[src*="hcaptcha.com"]')) {{{{
    const el = document.querySelector('[data-sitekey]');
    if (el) {{{{ r.type = 'hcaptcha'; r.sitekey = el.dataset.sitekey; }}}}
  }}}}
  // 2. Cloudflare Turnstile
  if (!r.type) {{{{
    const cf = document.querySelector('.cf-turnstile, [data-turnstile-sitekey]');
    if (cf) {{{{
      r.type = 'turnstile'; r.sitekey = cf.dataset.sitekey || cf.dataset.turnstileSitekey;
      if (cf.dataset.action) r.action = cf.dataset.action;
      if (cf.dataset.cdata) r.cdata = cf.dataset.cdata;
    }}}}
  }}}}
  if (!r.type && document.querySelector('script[src*="challenges.cloudflare.com"]')) {{{{
    r.type = 'turnstile_script_only'; r.note = 'Wait 3s and re-detect.';
  }}}}
  // 3. reCAPTCHA v3 (invisible, loaded via render= param)
  if (!r.type) {{{{
    const s = document.querySelector('script[src*="recaptcha"][src*="render="]');
    if (s) {{{{
      const m = s.src.match(/render=([^&]+)/);
      if (m && m[1] !== 'explicit') {{{{ r.type = 'recaptchav3'; r.sitekey = m[1]; }}}}
    }}}}
  }}}}
  // 4. reCAPTCHA v2 (checkbox or invisible)
  if (!r.type) {{{{
    const rc = document.querySelector('.g-recaptcha');
    if (rc) {{{{ r.type = 'recaptchav2'; r.sitekey = rc.dataset.sitekey; }}}}
  }}}}
  if (!r.type && document.querySelector('script[src*="recaptcha"]')) {{{{
    const el = document.querySelector('[data-sitekey]');
    if (el) {{{{ r.type = 'recaptchav2'; r.sitekey = el.dataset.sitekey; }}}}
  }}}}
  // 5. FunCaptcha (Arkose Labs)
  if (!r.type) {{{{
    const fc = document.querySelector('#FunCaptcha, [data-pkey], .funcaptcha');
    if (fc) {{{{ r.type = 'funcaptcha'; r.sitekey = fc.dataset.pkey; }}}}
  }}}}
  if (!r.type && document.querySelector('script[src*="arkoselabs"], script[src*="funcaptcha"]')) {{{{
    const el = document.querySelector('[data-pkey]');
    if (el) {{{{ r.type = 'funcaptcha'; r.sitekey = el.dataset.pkey; }}}}
  }}}}
  if (r.type) {{{{ r.url = url; return r; }}}}
  return null;
}}}}

Result actions:
- null -> no CAPTCHA. Continue normally.
- "turnstile_script_only" -> browser_wait_for time: 3, re-run detect.
- Any other type -> proceed to CAPTCHA SOLVE below.

--- CAPTCHA SOLVE ---
Three steps: createTask -> poll -> inject. Do each as a separate browser_evaluate call.

STEP 1 -- CREATE TASK (copy this exactly, fill in the 3 placeholders):
browser_evaluate function: async () => {{{{
  const r = await fetch('https://api.capsolver.com/createTask', {{{{
    method: 'POST',
    headers: {{{{'Content-Type': 'application/json'}}}},
    body: JSON.stringify({{{{
      clientKey: '{capsolver_key}',
      task: {{{{
        type: 'TASK_TYPE',
        websiteURL: 'PAGE_URL',
        websiteKey: 'SITE_KEY'
      }}}}
    }}}})
  }}}});
  return await r.json();
}}}}

TASK_TYPE values (use EXACTLY these strings):
  hcaptcha     -> HCaptchaEnterpriseTaskProxyLess
  recaptchav2  -> ReCaptchaV2TaskProxyLess
  recaptchav3  -> ReCaptchaV3TaskProxyLess
  turnstile    -> AntiTurnstileTaskProxyLess
  funcaptcha   -> FunCaptchaTaskProxyLess

PAGE_URL = the url from detect result. SITE_KEY = the sitekey from detect result.
For recaptchav3: add "pageAction": "submit" to the task object (or the actual action found in page scripts).
For turnstile: add "metadata": {{"action": "...", "cdata": "..."}} if those were in detect result.
For hcaptcha -- use this 3-step fallback chain. Stop as soon as one returns errorId == 0:
  1. HCaptchaEnterpriseTaskProxyLess (add "isInvisible": true and "enterprisePayload": {{"rqdata": RQ}} if rqdata was found)
  2. HCaptchaTaskProxyLess
  3. HCaptchaTurboTask (this handles the hardest enterprise hCaptchas)
  To find rqdata, run browser_evaluate BEFORE createTask:
    browser_evaluate function: () => {{{{ const c = document.querySelector('[data-sitekey]'); return c ? (c.dataset.rqdata || c.getAttribute('data-rqdata') || '') : ''; }}}}
  If rqdata is non-empty, include it in EVERY hcaptcha createTask attempt.

Response: {{"errorId": 0, "taskId": "abc123"}} on success.
If errorId > 0 after all fallback attempts -> CAPTCHA SOLVE failed. Go to MANUAL FALLBACK.

STEP 2 -- POLL (replace TASK_ID with the taskId from step 1):
Loop: browser_wait_for time: 3, then run:
browser_evaluate function: async () => {{{{
  const r = await fetch('https://api.capsolver.com/getTaskResult', {{{{
    method: 'POST',
    headers: {{{{'Content-Type': 'application/json'}}}},
    body: JSON.stringify({{{{
      clientKey: '{capsolver_key}',
      taskId: 'TASK_ID'
    }}}})
  }}}});
  return await r.json();
}}}}

- status "processing" -> wait 3s, poll again. Max 10 polls (30s).
- status "ready" -> extract token:
    reCAPTCHA: solution.gRecaptchaResponse
    hCaptcha:  solution.gRecaptchaResponse
    Turnstile: solution.token
- errorId > 0 or 30s timeout -> MANUAL FALLBACK.

STEP 3 -- INJECT TOKEN (replace THE_TOKEN with actual token string):

For reCAPTCHA v2/v3:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  // Suppress alert/confirm/prompt so stray callbacks can't spawn blocking dialogs
  const _a = window.alert, _c = window.confirm, _p = window.prompt;
  window.alert = () => {{{{}}}}; window.confirm = () => true; window.prompt = () => '';
  document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {{{{ el.value = token; el.style.display = 'block'; }}}});
  let callbackFired = false;
  // Method 1: walk grecaptcha clients and fire only short-named functions (safe)
  if (window.___grecaptcha_cfg) {{{{
    const clients = window.___grecaptcha_cfg.clients;
    for (const key in clients) {{{{
      const walk = (obj, d) => {{{{
        if (d > 5 || !obj) return;
        for (const k in obj) {{{{
          if (typeof obj[k] === 'function' && k.length < 3) try {{{{ obj[k](token); callbackFired = true; }}}} catch(e) {{{{}}}}
          else if (typeof obj[k] === 'object') walk(obj[k], d+1);
        }}}}
      }}}};
      walk(clients[key], 0);
    }}}}
  }}}}
  // Method 2: try grecaptcha.enterprise callback
  if (window.grecaptcha && window.grecaptcha.enterprise) {{{{
    try {{{{ window.grecaptcha.enterprise.execute && window.grecaptcha.enterprise.execute(); callbackFired = true; }}}} catch(e) {{{{}}}}
  }}}}
  // Method 3: find and call any global reCAPTCHA callback
  for (const k of Object.keys(window)) {{{{
    if (/recaptcha|captcha|onsubmit|verify/i.test(k) && typeof window[k] === 'function') {{{{
      try {{{{ window[k](token); callbackFired = true; }}}} catch(e) {{{{}}}}
    }}}}
  }}}}
  // Method 4: dispatch events on the response textarea to trigger framework listeners
  document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {{{{
    el.dispatchEvent(new Event('input', {{{{bubbles: true}}}}));
    el.dispatchEvent(new Event('change', {{{{bubbles: true}}}}));
  }}}});
  // Restore originals
  window.alert = _a; window.confirm = _c; window.prompt = _p;
  return callbackFired ? 'injected+callback' : 'injected_no_callback';
}}}}

For hCaptcha:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  const _a = window.alert, _c = window.confirm, _p = window.prompt;
  window.alert = () => {{{{}}}}; window.confirm = () => true; window.prompt = () => '';
  const ta = document.querySelector('[name="h-captcha-response"], textarea[name*="hcaptcha"]');
  if (ta) ta.value = token;
  document.querySelectorAll('iframe[data-hcaptcha-response]').forEach(f => f.setAttribute('data-hcaptcha-response', token));
  const cb = document.querySelector('[data-hcaptcha-widget-id]');
  if (cb && window.hcaptcha) try {{{{ window.hcaptcha.getResponse(cb.dataset.hcaptchaWidgetId); }}}} catch(e) {{{{}}}}
  window.alert = _a; window.confirm = _c; window.prompt = _p;
  return 'injected';
}}}}

For Turnstile:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  const inp = document.querySelector('[name="cf-turnstile-response"], input[name*="turnstile"]');
  if (inp) inp.value = token;
  if (window.turnstile) try {{{{ const w = document.querySelector('.cf-turnstile'); if (w) window.turnstile.getResponse(w); }}}} catch(e) {{{{}}}}
  return 'injected';
}}}}

For FunCaptcha:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  const _a = window.alert, _c = window.confirm, _p = window.prompt;
  window.alert = () => {{{{}}}}; window.confirm = () => true; window.prompt = () => '';
  const inp = document.querySelector('#FunCaptcha-Token, input[name="fc-token"]');
  if (inp) inp.value = token;
  if (window.ArkoseEnforcement) try {{{{ window.ArkoseEnforcement.setConfig({{{{data: {{{{blob: token}}}}}}}}) }}}} catch(e) {{{{}}}}
  window.alert = _a; window.confirm = _c; window.prompt = _p;
  return 'injected';
}}}}

After injecting: browser_wait_for time: 2, then browser_snapshot.
CRITICAL POST-INJECTION STEPS:
1. After the snapshot, IMMEDIATELY find and click the Submit/Apply/Continue button. Do NOT wait for auto-submit — you MUST click the button yourself.
2. If the page shows a "thank you" or "application received" message -> RESULT:APPLIED
3. If clicking Submit triggers another CAPTCHA -> re-run CAPTCHA DETECT and solve again.
4. If the Submit button is disabled after injection, try ONE more time: re-run CAPTCHA SOLVE from STEP 1 with a fresh token, inject, and try Submit again.
5. If the Submit button is STILL disabled after the second injection attempt -> GIVE UP. Output RESULT:CAPTCHA immediately. Do NOT keep retrying.
6. NEVER call browser_wait_for more than twice in a row. If nothing changed after 2 waits, take action or give up.
7. NEVER spend more than 30 seconds total on CAPTCHA solving. If you've been working on CAPTCHA for ~30s with no progress, output RESULT:CAPTCHA and move on.
Do NOT get stuck in a wait loop. The moment you realize CAPTCHA isn't working, output RESULT:CAPTCHA so we can move to the next job.

--- MANUAL FALLBACK ---
You should ONLY be here if CapSolver createTask returned errorId > 0. If you haven't tried CapSolver yet, GO BACK and try it first.
Reminder: NEVER click the reCAPTCHA/hCaptcha checkbox. NEVER attempt audio challenges or image selection. These are unsolvable for you.
If CapSolver genuinely failed (errorId > 0):
1. Simple text captchas ("What is 3+7?", "Type the word") -> solve them.
2. Text/logic puzzles: Solve them yourself. Think step by step.
3. All else fails -> Output RESULT:CAPTCHA."""


def build_prompt(job: dict, tailored_resume: str,
                 cover_letter: str | None = None,
                 dry_run: bool = False) -> str:
    """Build the full instruction prompt for the apply agent.

    Loads the user profile and search config internally. All personal data
    comes from the profile -- nothing is hardcoded.

    Args:
        job: Job dict from the database (must have url, title, site,
             application_url, fit_score, tailored_resume_path).
        tailored_resume: Plain-text content of the tailored resume.
        cover_letter: Optional plain-text cover letter content.
        dry_run: If True, tell the agent not to click Submit.

    Returns:
        Complete prompt string for the AI agent.
    """
    profile = config.load_profile()
    search_config = config.load_search_config()
    personal = profile["personal"]

    # --- Resolve resume PDF path ---
    resume_path = job.get("tailored_resume_path")
    if not resume_path:
        raise ValueError(f"No tailored resume for job: {job.get('title', 'unknown')}")

    src_pdf = Path(resume_path).with_suffix(".pdf").resolve()
    if not src_pdf.exists():
        raise ValueError(f"Resume PDF not found: {src_pdf}")

    # Copy to a clean filename for upload (recruiters see the filename)
    full_name = personal["full_name"]
    name_slug = full_name.replace(" ", "_")
    dest_dir = config.APPLY_WORKER_DIR / "current"
    dest_dir.mkdir(parents=True, exist_ok=True)
    upload_pdf = dest_dir / f"{name_slug}_Resume.pdf"
    shutil.copy(str(src_pdf), str(upload_pdf))
    pdf_path = str(upload_pdf)

    # --- Cover letter handling ---
    cover_letter_text = cover_letter or ""
    cl_upload_path = ""
    cl_path = job.get("cover_letter_path")
    if cl_path and Path(cl_path).exists():
        cl_src = Path(cl_path)
        # Read text from .txt sibling (PDF is binary)
        cl_txt = cl_src.with_suffix(".txt")
        if cl_txt.exists():
            cover_letter_text = cl_txt.read_text(encoding="utf-8")
        elif cl_src.suffix == ".txt":
            cover_letter_text = cl_src.read_text(encoding="utf-8")
        # Upload must be PDF
        cl_pdf_src = cl_src.with_suffix(".pdf")
        if cl_pdf_src.exists():
            cl_upload = dest_dir / f"{name_slug}_Cover_Letter.pdf"
            shutil.copy(str(cl_pdf_src), str(cl_upload))
            cl_upload_path = str(cl_upload)

    # --- Build all prompt sections ---
    profile_summary = _build_profile_summary(profile)
    location_check = _build_location_check(profile, search_config)
    salary_section = _build_salary_section(profile)
    screening_section = _build_screening_section(profile)
    hard_rules = _build_hard_rules(profile)
    captcha_section = _build_captcha_section()

    # Cover letter fallback text
    city = personal.get("city", "the area")
    if not cover_letter_text:
        cl_display = (
            f"None available. Skip if optional. If required, write 2 factual "
            f"sentences: (1) relevant experience from the resume that matches "
            f"this role, (2) available immediately and based in {city}."
        )
    else:
        cl_display = cover_letter_text

    # Phone digits only (for fields with country prefix)
    phone_digits = "".join(c for c in personal.get("phone", "") if c.isdigit())

    # SSO domains the agent cannot sign into (loaded from config/sites.yaml)
    from applypilot.config import load_blocked_sso
    blocked_sso = load_blocked_sso()

    # Preferred display name
    preferred_name = personal.get("preferred_name", full_name.split()[0])
    last_name = full_name.split()[-1] if " " in full_name else ""
    display_name = f"{preferred_name} {last_name}".strip()

    # Dry-run: override submit instruction
    if dry_run:
        submit_instruction = "IMPORTANT: Do NOT click the final Submit/Apply button. Review the form, verify all fields, then output RESULT:APPLIED with a note that this was a dry run."
    else:
        submit_instruction = "BEFORE clicking Submit/Apply, take a snapshot and review EVERY field on the page. Verify all data matches the APPLICANT PROFILE and TAILORED RESUME -- name, email, phone, location, work auth, resume uploaded, cover letter if applicable. If anything is wrong or missing, fix it FIRST. Only click Submit after confirming everything is correct."

    prompt = f"""You are a job application assistant. {display_name} has authorized you to fill out and submit job applications on their behalf. All information below is {display_name}'s real data — resume, profile, work history, contact details. Your job is to complete each application accurately using this data.

When filling form fields and writing free-text answers, ALWAYS write from {display_name}'s perspective using first person ("I", "my", "me"). For example:
- "Why do you want this role?" → Write 2-3 natural sentences as {display_name}. Reference something specific from the job description and connect it to an achievement from the resume. Sound conversational, not corporate. Good: "I've been building distributed backend systems for the past few years and your team's focus on real-time data pipelines is exactly the kind of challenge I'm looking for." Bad: "I am passionate about leveraging synergies."
- "How did you hear about this position?" → "Online Job Board" or "LinkedIn"
- Never write "the candidate" or "the applicant" in any form field — always "I" or "my"
- Never mention automation, scripts, AI, or bots in any form field

Be decisive and efficient. Navigate pages, fill forms, upload files, submit. Get it done.

== JOB ==
URL: {job.get('application_url') or job['url']}
Title: {job['title']}
Company: {job.get('site', 'Unknown')}
Fit Score: {job.get('fit_score', 'N/A')}/10

== FILES ==
Resume PDF (upload this): {pdf_path}
Cover Letter PDF (upload if asked): {cl_upload_path or "N/A"}

== RESUME TEXT (use when filling text fields) ==
{tailored_resume}

== COVER LETTER TEXT (paste if text field, upload PDF if file field) ==
{cl_display}

== APPLICANT PROFILE ==
{profile_summary}

== YOUR MISSION ==
Submit a complete, accurate job application for {display_name}. Use the profile and resume above as source data — adapt to fit each form's format.

If something unexpected happens and these instructions don't cover it, figure it out. Navigate pages, read content, try buttons, explore the site. The goal is always the same: get the application submitted. Do whatever it takes to reach that goal.

CRITICAL: You are fully autonomous. NEVER ask questions or wait for user input. NEVER say "How would you like to proceed?" or present options. Make your own decisions and keep going. If you get redirected to a different page, navigate back to the original URL. If something goes wrong, try to fix it yourself. If you truly cannot proceed, output a RESULT code immediately.

{hard_rules}

== NEVER DO THESE (immediate RESULT:FAILED if encountered) ==
- NEVER grant camera, microphone, screen sharing, or location permissions. If a site requests them -> RESULT:FAILED:unsafe_permissions
- NEVER do video/audio verification, selfie capture, ID photo upload, or biometric anything -> RESULT:FAILED:unsafe_verification
- NEVER set up a freelancing profile (Mercor, Toptal, Upwork, Fiverr, Turing, etc.). These are contractor marketplaces, not job applications -> RESULT:FAILED:not_a_job_application
- NEVER agree to hourly/contract rates, availability calendars, or "set your rate" flows. You are applying for FULL-TIME salaried positions only.
- NEVER install browser extensions, download executables, or run assessment software.
- NEVER enter payment info, bank details, or SSN/SIN.
- NEVER click "Allow" on any browser permission popup. Always deny/block.
- If the site is NOT a job application form (it's a profile builder, skills marketplace, talent network signup, coding assessment platform) -> RESULT:FAILED:not_a_job_application

{location_check}

{salary_section}

{screening_section}

== STEP-BY-STEP ==
1. browser_navigate to the job URL.
2. browser_snapshot to read the page. Then run CAPTCHA DETECT (see CAPTCHA section). If a CAPTCHA is found, solve it before continuing.
3. Find and click the Apply button. If email-only (page says "email resume to X"):
   - Use the send_email MCP tool: to = the email address shown, subject = "Application for {job['title']} – {personal['full_name']}", body = the cover letter text + "Please find my resume attached.", attachment = the resume PDF path above.
   - After sending, output RESULT:APPLIED.
   After clicking Apply: browser_snapshot. Run CAPTCHA DETECT -- many sites trigger CAPTCHAs right after the Apply click. If found, solve before continuing.
4. Login wall?
   4a. FIRST: check the URL. If you landed on {', '.join(blocked_sso)}, or any SSO/OAuth page -> STOP. Output RESULT:FAILED:sso_required. Do NOT try to sign in to Google/Microsoft/SSO.
   4b. Check for popups. Run browser_tabs action "list". If a new tab/window appeared (login popup), switch to it with browser_tabs action "select". Check the URL there too -- if it's SSO -> RESULT:FAILED:sso_required.
     4c. IMPORTANT — ATS portal detection. If the URL contains any of these: myworkdayjobs.com, myworkday.com, workday.com, taleo, icims, successfactors, sapsf.com, sap.com/careers, brassring — this is an EMPLOYER-SPECIFIC ATS portal. Your personal passwords will NOT work here because each employer has their own user database. SKIP sign-in entirely and go STRAIGHT to "Create Account" / "New User" / "Register". Use {personal['email']} / {personal.get('password', '')} for the new account.
       AFTER creating account: sign in with THE EXACT SAME email and password you just used to create the account. Do NOT try other passwords. If sign-in fails after account creation (e.g., site asks for email verification), use search_emails + read_email (Gmail MCP) to get the code or link.
   4d. NON-ATS login form (Indeed, company career site, BambooHR, Lever, Greenhouse, etc.)? Try these credentials QUICKLY:
       - {personal['email']} / {personal.get('password', '')}
       *** CRITICAL: NEVER reset passwords on LinkedIn (linkedin.com) or Gmail (google.com/gmail.com). If login fails on LinkedIn or Gmail, output RESULT:FAILED:login_issue immediately. For ALL OTHER sites (career portals, ATS, job boards, everything else), you MAY use the "Forgot Password" flow below. ***
       If BOTH passwords fail AND the site is NOT LinkedIn or Gmail, use FORGOT PASSWORD:
       1. Click "Forgot Password" / "Reset Password" link on the login page.
       2. Enter {personal['email']} and submit the reset request.
       3. Use search_emails + read_email (Gmail MCP) to find the password reset email, then click the reset link via browser_navigate.
       4. Set the new password to: {personal.get('password', '')}
       5. Go back to the login page and sign in with {personal['email']} / {personal.get('password', '')}.
       If the site has no account for {personal['email']}, create one with {personal['email']} / {personal.get('password', '')}.
   4d2. After clicking Login/Sign-in: run CAPTCHA DETECT. Login pages frequently have invisible CAPTCHAs that silently block form submissions. If found, solve it then retry login.
   4e. IMPORTANT: Do NOT loop between sign-in and sign-up more than once. If you've tried sign-in -> forgot password -> sign-in again and it still fails, output RESULT:FAILED:login_issue immediately.
   4f. Email verification required? Use the Gmail MCP tools:
       1. search_emails with query "is:unread newer_than:5m" to find the verification email.
       2. read_email on the matching email to get the verification code or link.
       3. If it's a CODE: enter it on the application page.
       4. If it's a LINK: browser_navigate to the link, complete verification, then go back to the application.
       5. No email found? Wait 10 seconds, then search_emails again. Try up to 3 times.
       Only output RESULT:FAILED:email_verification if search_emails returns nothing after 3 attempts.
   4g. After login, run browser_tabs action "list" again. Switch back to the application tab if needed.
   4h. All login attempts failed (including email verification)? Output RESULT:FAILED:login_issue. Do not loop.
5. Upload resume. ALWAYS upload fresh -- delete any existing resume first, then browser_file_upload with the PDF path above. This is the tailored resume for THIS job. Non-negotiable.
6. Upload cover letter if there's a field for it. Text field -> paste the cover letter text. File upload -> use the cover letter PDF path.
7. Check ALL pre-filled fields. ATS systems parse your resume and auto-fill -- it's often WRONG.
   - "Current Job Title" or "Most Recent Title" -> use the title from the TAILORED RESUME summary, NOT whatever the parser guessed.
   - Compare every other field to the APPLICANT PROFILE. Fix mismatches. Fill empty fields.
8. Answer screening questions using the rules above.
9. {submit_instruction}
10. After submit: browser_snapshot. Run CAPTCHA DETECT -- submit buttons often trigger invisible CAPTCHAs. If found, solve it (the form will auto-submit once the token clears, or you may need to click Submit again). Then check for new tabs (browser_tabs action: "list"). Switch to newest, close old. Snapshot to confirm submission. Look for "thank you" or "application received".
11. Output your result.

== RESULT CODES (output EXACTLY one of these — this is MANDATORY) ==
RESULT:APPLIED -- submitted successfully
RESULT:EXPIRED -- job closed or no longer accepting applications
RESULT:CAPTCHA -- blocked by unsolvable captcha AFTER multiple solve attempts
RESULT:LOGIN_ISSUE -- could not sign in or create account (AFTER trying email verification via Gmail MCP)
RESULT:FAILED:not_eligible_work_auth -- requires unauthorized work location
RESULT:FAILED:reason -- any other failure (brief reason)

You MUST output exactly one RESULT: line. This is non-negotiable. Never end without a RESULT code.
NEVER ask the user questions. NEVER say "how would you like to proceed". Just DO it.

== SPEED (critical — you have a 4 minute hard deadline) ==
- You MUST complete each application in under 4 minutes. Every second counts.
- browser_snapshot ONCE per page. After that use browser_take_screenshot (10x faster).
- Fill ALL fields in ONE browser_fill_form call. NEVER fill fields one at a time.
- Keep thinking SHORT. No re-describing the page. Just act.
- If a page takes >10s to load, skip it: RESULT:FAILED:page_load_timeout
- If login takes >2 attempts, give up: RESULT:FAILED:login_issue
- If CAPTCHA solving takes >30s, give up: RESULT:CAPTCHA
- CAPTCHA AWARENESS: After clicking Apply/Submit/Login, run CAPTCHA DETECT. Invisible CAPTCHAs block submissions silently.

== FORM TRICKS ==
- New tab/popup? browser_tabs "list" then "select" to switch.
- Upload page (Workday, Lever): click upload area, browser_file_upload, wait for parse, click Next.
- Dropdown stuck? browser_click to open, browser_click the option.
- Checkbox stuck? browser_click instead of fill_form.
- Phone: digits only {phone_digits}. Date: {datetime.now().strftime('%m/%d/%Y')}
- Validation errors? Screenshot + fix + retry once. If still failing, submit anyway.

{captcha_section}

== EMAIL VERIFICATION (via Gmail MCP) ==
You have Gmail MCP tools: search_emails and read_email. Use them — do NOT navigate to Gmail in the browser.

When ANY site requires email verification, a confirmation code, or a magic link:
1. search_emails with query: "is:unread newer_than:5m" (finds all unread emails from last 5 minutes).
2. Look for a verification/confirmation email from the site you're applying to.
3. read_email on that email's ID to get the full body.
4. Extract the verification code (usually 4-8 digits) or the verification/confirmation link.
5. If it's a CODE: enter it on the application page.
6. If it's a LINK: browser_navigate to the link. After verification completes, browser_navigate back to the application URL.
7. If no matching email found: wait 10 seconds, then search_emails again. Try up to 3 times total.
8. If still no email after 3 attempts: output RESULT:FAILED:email_verification.

Do NOT navigate to Gmail in the browser. Use the MCP tools — they are faster and more reliable.

== WORKDAY TIPS ==
Workday portals (myworkdayjobs.com) are multi-page. Key tips:
- After the initial job page, click "Apply" or "Apply Manually". You may land on a sign-in page.
- For sign-in: ALWAYS try "Create Account" first. Use {personal['email']} / {personal.get('password', '')}. If already registered, sign in with those credentials. If the password doesn't work, use "Forgot Password", use search_emails + read_email (Gmail MCP) to get the reset link, set password to {personal.get('password', '')}, then sign in.
- Workday often requires email verification after account creation. Use search_emails + read_email (Gmail MCP) to get the verification code.
- After login, you'll see a multi-step form (My Information -> My Experience -> Application Questions -> Review -> Submit). Fill each page, click "Next" or "Save and Continue".
- "My Experience" page: upload the resume PDF. Workday will auto-parse it — CHECK and FIX all parsed fields.
- "How Did You Hear About Us" is usually required. Answer "Online Job Board" or "LinkedIn".
- On the final Review page: verify everything, then click Submit.
- If Workday shows "You have already applied" or similar -> output RESULT:FAILED:already_applied.

== WHEN TO GIVE UP ==
- Same page after 3 attempts with no progress -> RESULT:FAILED:stuck
- Job is closed/expired/page says "no longer accepting" -> RESULT:EXPIRED
- Page is broken/500 error/blank -> RESULT:FAILED:page_error
Stop immediately. Output your RESULT code. Do not loop."""

    return prompt
