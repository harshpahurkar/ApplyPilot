# ApplyPilot Execution Brief (Organized)

## Mission
Run ApplyPilot end-to-end with strict ATS-quality tailoring, Canada/remote boundary enforcement, and verified real applications (not false positives), while avoiding Copilot rate-limit lockouts.

## Priority Order
1. Reconstruct and report what changed in previous runs and code updates.
2. Ensure all pending workspace changes are applied and runnable.
3. Enforce strict ATS tailoring quality (not speed-first shortcuts).
4. Prevent Copilot rate-limit lockouts with safer request pacing.
5. Prove real application outcomes from logs + database.
6. Verify Gmail MCP is actually working for verification/email checks.
7. Explain why jobs were skipped and how to reduce bad skips.
8. Run a small visible apply batch with headless disabled so I can manually observe behavior.

## Hard Constraints
- Geography:
  - Allow: jobs in Canada.
  - Allow: global remote only when posting does not restrict to non-Canada geographies.
  - Disallow: roles explicitly US-only, India-only, or otherwise non-Canada restricted.
- Tailoring quality:
  - Strict ATS optimization required.
  - No weak/generic tailoring.
  - Resume bullets must remain measurable, role-aligned, and keyword-accurate.
- Truthfulness:
  - Do not claim success without evidence in DB/logs.
  - Separate "applied", "attempted", "failed", "skipped", and "expired" clearly.

## Strict ATS Tailoring Rules (Ball Knowledge)
Use this workflow for each target role:
1. Gap analysis vs job description:
   - Match score (1-10) with reasoning.
   - Top missing keywords/phrases ranked by ATS importance.
   - Skills and terminology gaps.
   - Identify JD's top 5 most-mentioned keywords for frequency targeting.
2. Experience rewrite requirements:
   - Bullet pattern: Achieved X by Y resulting in Z.
   - 2-3 quantified achievements per role.
   - Strong action verbs and role-specific language.
   - Natural keyword integration (no stuffing).
   - Max 130 characters per bullet (no orphan line overflow).
   - Never end a bullet with a tech list (use skills_used field).
   - Every bullet must have at least one specific number.
   - JD "required" skills must appear in experience (not just Technical Skills) to get years-of-experience credit from ATS.
   - Top 5 JD keywords must each appear 2-3 times across the resume.
   - Acronyms must include full form at least once (e.g., "CI/CD (Continuous Integration/Continuous Deployment)").
   - Weave top 3 JD soft skills naturally into experience bullets.
3. ATS scan requirements:
   - Parsing-safe formatting.
   - Missing/overused keyword detection.
   - Header/date/contact compatibility checks.
   - Adapt coursework field to echo JD keywords (59.7% of recruiters filter by education).
   - Every bullet must reference a specific tool/system/metric (anti-generic check — 28% reject AI-sounding content).
4. Output quality bar:
   - Target ATS score >= 90.
   - Role match score >= 8/10.
   - No graphics/tables parsing risks.
   - 18-point pre-output verification checklist (up from 14).
5. Mandatory 2-iteration minimum:
   - Every resume goes through at least 2 LLM iterations.
   - First pass generates initial resume. Second pass refines with specific feedback on keyword density, bullet formula compliance, and skill placement.
   - Even if the first pass scores 100%, the second pass still runs to catch subtle improvements.

## Copilot Rate-Limit Mitigation Requirements
- Treat rate limit as a hard operational constraint.
- Use conservative pacing and bounded concurrency.
- Avoid bursty parallel Copilot requests across many workers.
- Prefer queued/serialized Copilot-heavy steps when needed.
- Add adaptive cooldown handling when response contains "try again in ...".
- Keep throughput just below limit instead of maximizing spikes.

## Architecture Preference Request
- Investigate feasibility of minimizing short-lived Copilot chat session churn.
- If possible, prefer fewer long-lived interaction channels over many rapid session starts.
- Use direct API-style calls where available for scoring/tailoring steps, if this reduces session churn and preserves quality.

## Gmail MCP Verification Requirements
- Confirm auth health (token validity, credential freshness).
- Confirm tools are callable and return expected email search/read results.
- Distinguish password-reset emails vs application-success emails.
- If broken, run re-auth and re-test until verified.

## Run Verification Requirements (Evidence)
For each run, provide:
- Counts: discovered, shortlisted, attempted, applied, skipped, failed, expired.
- Top skip reasons with frequency.
- Top fail reasons with frequency.
- Recent applied job records (role/company/time/source).
- Log snippets proving Gmail MCP activity and apply result outcomes.

## Immediate Execution Plan
1. Health-check environment and pending code changes.
2. Validate strict ATS rules are active in tailoring + validator paths.
3. Validate location filtering logic matches Canada/remote policy.
4. Apply Copilot pacing/concurrency safeguards.
5. Validate Gmail MCP auth + test search/read.
6. Run a small headless=false apply batch for manual observation.
7. Produce evidence report with hard numbers and reason breakdowns.

## Definition of Done
- Strict ATS behavior confirmed in code and observed outputs.
- Copilot rate-limit incidents reduced/controlled during run.
- Gmail MCP verified functional in real workflow.
- At least one new, clearly evidenced real application outcome OR a precise blocker report with exact root cause and next fix.
- Skip/fail reason analysis provided with actionable fixes.
