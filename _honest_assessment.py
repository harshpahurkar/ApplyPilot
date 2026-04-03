import sqlite3, os
db = sqlite3.connect(os.path.expanduser("~/.applypilot/applypilot.db"))

print("=== SUCCESS RATE BY PORTAL ===")
for r in db.execute("""
    SELECT 
        CASE 
            WHEN COALESCE(application_url, url) LIKE '%greenhouse%' OR COALESCE(application_url, url) LIKE '%grnh.se%' THEN 'greenhouse'
            WHEN COALESCE(application_url, url) LIKE '%ashby%' THEN 'ashby'
            WHEN COALESCE(application_url, url) LIKE '%rippling%' THEN 'rippling'
            WHEN COALESCE(application_url, url) LIKE '%workable%' THEN 'workable'
            WHEN COALESCE(application_url, url) LIKE '%smartrecruiters%' THEN 'smartrecruiters'
            WHEN COALESCE(application_url, url) LIKE '%betterteam%' THEN 'betterteam'
            WHEN COALESCE(application_url, url) LIKE '%indeed%' THEN 'indeed'
            WHEN COALESCE(application_url, url) LIKE '%workday%' OR COALESCE(application_url, url) LIKE '%myworkdayjobs%' THEN 'workday'
            ELSE 'other'
        END as portal,
        SUM(CASE WHEN apply_status = 'applied' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN apply_status = 'failed' THEN 1 ELSE 0 END) as losses
    FROM jobs 
    WHERE apply_status IN ('applied', 'failed')
    GROUP BY portal
    ORDER BY wins DESC
"""):
    total = r[1] + r[2]
    pct = (100.0 * r[1] / total) if total else 0
    print(f"  {r[0]}: {r[1]}W / {r[2]}L = {pct:.0f}%")

print()
for r in db.execute("SELECT ROUND(AVG(apply_duration_ms)/1000.0, 0) FROM jobs WHERE apply_status = 'applied' AND apply_duration_ms > 0"):
    print(f"Avg time per successful apply: {r[0]}s")
for r in db.execute("SELECT ROUND(SUM(apply_duration_ms)/1000.0/60.0, 1) FROM jobs WHERE apply_status = 'failed'"):
    print(f"Total time wasted on failures: {r[0]} min")
for r in db.execute("SELECT ROUND(SUM(apply_duration_ms)/1000.0/60.0, 1) FROM jobs WHERE apply_status = 'applied'"):
    print(f"Total time on successes: {r[0]} min")

print("\n=== SUPPLY OF GOOD-PORTAL JOBS ===")
for r in db.execute("""
    SELECT 
        CASE 
            WHEN COALESCE(application_url, url) LIKE '%greenhouse%' OR COALESCE(application_url, url) LIKE '%grnh.se%' THEN 'greenhouse'
            WHEN COALESCE(application_url, url) LIKE '%ashby%' THEN 'ashby'
            WHEN COALESCE(application_url, url) LIKE '%smartrecruiters%' THEN 'smartrecruiters'
            WHEN COALESCE(application_url, url) LIKE '%rippling%' THEN 'rippling'
            WHEN COALESCE(application_url, url) LIKE '%workable%' THEN 'workable'
            ELSE NULL
        END as portal,
        SUM(CASE WHEN tailored_resume_path IS NOT NULL AND apply_status IS NULL THEN 1 ELSE 0 END) as ready,
        SUM(CASE WHEN fit_score IS NOT NULL AND tailored_resume_path IS NULL AND apply_status IS NULL THEN 1 ELSE 0 END) as needs_tailor,
        SUM(CASE WHEN fit_score IS NULL AND apply_status IS NULL THEN 1 ELSE 0 END) as needs_score
    FROM jobs 
    WHERE apply_status IS NULL
    GROUP BY portal
    HAVING portal IS NOT NULL
    ORDER BY (ready + needs_tailor + needs_score) DESC
"""):
    total = r[1] + r[2] + r[3]
    print(f"  {r[0]}: {total} total ({r[1]} ready, {r[2]} need tailoring, {r[3]} need scoring)")
