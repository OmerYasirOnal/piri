# Staging Screenshots - Status Report

## Summary

Successfully captured screenshots from `staging.akisflow.com` for the Piri public repository README.

## Available Screenshots

### ✅ Captured Screenshots

1. **`agent-hub.png`** (2880 x 1800)
   - Agent Hub / Chat interface
   - Shows agent selection and conversation UI
   - Location: `/Users/omeryasironal/Projects/bitirme_projesi/akis-platform-devolopment/devagents/piri/docs/screenshots/agent-hub.png`

2. **`job-detail.png`** (2880 x 1800)
   - Job detail page
   - Shows trace events and job information
   - Location: `/Users/omeryasironal/Projects/bitirme_projesi/akis-platform-devolopment/devagents/piri/docs/screenshots/job-detail.png`

3. **`dashboard.png`** (2880 x 1800)
   - Dashboard overview
   - Location: `/Users/omeryasironal/Projects/bitirme_projesi/akis-platform-devolopment/devagents/piri/docs/screenshots/dashboard.png`

4. **`main-page.png`** (3840 x 13562)
   - Landing/home page (full page)
   - Location: `/Users/omeryasironal/Projects/bitirme_projesi/akis-platform-devolopment/devagents/piri/docs/screenshots/main-page.png`

5. **`login-page.png`** (3840 x 13562)
   - Login page (full page)
   - Location: `/Users/omeryasironal/Projects/bitirme_projesi/akis-platform-devolopment/devagents/piri/docs/screenshots/login-page.png`

### ⚠️ Pending Screenshot

- **`quality-score.png`** - Quality score visualization
  - Requires authenticated session and a completed job with quality metrics
  - Can be captured using `scripts/capture-screenshots.mjs` when logged in

## Scripts Created

### 1. `scripts/test-staging.mjs`
Full automated test script that:
- Tests `/health` endpoint
- Navigates through main pages
- Attempts login (requires manual intervention for multi-step auth)
- Captures screenshots at each step

### 2. `scripts/capture-screenshots.mjs`
Focused screenshot capture script that:
- Uses existing browser session/cookies
- Prompts for manual login if needed
- Captures agent hub, job details, and quality scores
- More reliable for authenticated screenshots

## Usage

### For Automated Testing
```bash
node scripts/test-staging.mjs
```

### For Manual Screenshot Capture (Recommended)
```bash
node scripts/capture-screenshots.mjs
```
The script will:
1. Open a browser window
2. Navigate to staging.akisflow.com/agents
3. Prompt you to login if needed
4. Capture screenshots after you press Enter

### For Quality Score Screenshot
1. Run `node scripts/capture-screenshots.mjs`
2. Login manually when prompted
3. Navigate to a completed job that shows quality scores
4. Press Enter to capture screenshots

## Next Steps

To complete the screenshot collection for Piri README:

1. **Option A - Manual Capture:**
   - Open staging.akisflow.com in browser
   - Login with your credentials
   - Navigate to a completed job showing quality scores
   - Take screenshot manually
   - Save to `piri/docs/screenshots/quality-score.png`

2. **Option B - Script with Manual Login:**
   - Run `node scripts/capture-screenshots.mjs`
   - Login when prompted
   - Let script auto-capture quality score section

## Technical Notes

- Health endpoint: ✅ Working (`{"status":"ok"}`)
- Main page: ✅ Loads successfully
- Login page: ✅ Renders correctly
- Auth flow: Multi-step (email → password → verification)
- Screenshots use 2x device scale factor (Retina quality)
- Puppeteer browser automation via puppeteer@24.37.2

## Staging Access Verified

- URL: https://staging.akisflow.com
- Health: 200 OK
- Response time: ~300ms
- Last tested: 2026-02-13 06:33 UTC
