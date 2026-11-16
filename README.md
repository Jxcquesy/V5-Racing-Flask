# V5 Racing Predictor — Flask / Render

This is a **separate Flask rebuild** of the uploaded V5 Streamlit repository. The supplied `v5_engine.py`, feature pipeline, PDF parser, and frozen `v5_top3_model.joblib` are reused unchanged. V4 is not involved.

## Deploy to Render

1. Create a **new GitHub repository** for these files (do not overwrite V4 or the current V5 Streamlit repo). Upload the contents of this folder to its root.
2. In Render choose **New → Blueprint** and connect the new repository. `render.yaml` configures a Python web service. Or create a Web Service manually with `pip install -r requirements.txt` and `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app`.
3. In Render **Environment** set `APP_PASSWORD` to a strong private login password, `SUPABASE_URL` to your existing Supabase project URL, and `SUPABASE_SERVICE_ROLE_KEY` to your existing service-role key. `SECRET_KEY` is generated automatically by the Blueprint; for manual deployments add a long random secret. **Never commit these values to GitHub.**
4. Set Python 3.12 where Render allows selecting it. Log in with `APP_PASSWORD`, open History, and confirm the old records are visible.
5. Upload your existing historical Master CSV on Data & Training. It was **not included** in the supplied ZIP. Upload a corrected racecard CSV or PDFs to run predictions.

## Important limitations

- **Master CSV storage is temporary** on Render's free instance filesystem. It can be lost after restart/redeploy. Download a backup after every merge. Permanent master storage needs a database or object storage integration.
- A separate daily V5 retraining runner is being introduced via GitHub Actions. It trains the existing model family using the existing feature definitions, from 1 January 2025 up to (but not including) the current UK race date, then publishes the model and status to the existing private Supabase bucket. The runner is separate from Render. It now retrains both the existing Top 3 model and a separate first-place Winner model, then publishes a chronological 60-day Winner backtest (up to 10 selections per day, one per race). Backtest metrics and selection CSV are available on the Data & Training page after the workflow runs. See `DAILY_TRAINING_SETUP.md` for setup. Automatic results fetching is not implemented.
- The PDF parser is best-effort and is reused unchanged. Unlike the Streamlit app, this Flask version currently has **no editable racecard verification grid**; upload a corrected CSV if the PDF extraction needs corrections.
- Supabase betting history uses the existing `v5_bets` table and compatible fields. No migration or history re-entry is needed. The £/unit setting is used for display but is not stored because the supplied Supabase schema does not include a unit-value column.
- The app uses password protection and server-side Supabase credentials. The login password should be private; for multi-user access add proper account-based authentication.
- The model is loaded from a trusted, owner-supplied joblib file. Do not replace it with an untrusted file.
- Render free web services may sleep when idle, and a paid plan may be required for more reliable availability.

## V5.1 experimental feature testing

The daily GitHub Actions workflow runs `train_v51_experiments.py` after normal V5 training. It compares the unchanged 42-feature baseline and four controlled feature-group ablations on the same chronological 60-day holdout. It also trains specialist models for recent form, ratings/class/distance, horse-and-connections history, and race/course/conditions.

Six explicit probability-blend profiles are tested: balanced, recent-form-led, ratings-led, horse/connections-led, race-conditions-led, and equal specialist weights. Each profile blends specialist-model probabilities with the full V5 baseline. This is a meaningful way to test group weights for a tree model; simply multiplying input columns would not reliably change their influence. The report includes Brier score, log loss, ROC AUC, daily Top-3 strike rate, selection count, and the exact weights used.

**Option 11 — Defiant Punter (DP) filter:** a rule-based shortlist, evaluated separately using the unchanged V5 baseline probabilities. A runner must (1) be dropping in class versus its previous run, (2) have finished in the top 20% of the field on its most recent previous run in today's class, (3) carry a weight in the lowest 50% of the current race, (4) not be among the three oldest runners, and (5) have previously run today's exact distance. The report records the eligible runner count, daily selections and Top-3 strike rate. This filter is not a probability model and is not promoted automatically.

A separate experimental model, JSON report, and comparison CSV are published under `models/v5_top3_v51_*`. The live Flask app continues to load only `models/v5_top3_model.joblib`; V5.1 is not used for live selections and cannot overwrite production V5. The same holdout is used to compare candidates, so the shortlisted variant must undergo later untouched-period shadow testing before any production promotion. No betting ROI is claimed because matched pre-race odds are unavailable.

## Local test

```bash
python -m venv .venv
# activate the virtual environment
pip install -r requirements.txt
# set SECRET_KEY, APP_PASSWORD, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in your shell
python app.py
```

Visit http://127.0.0.1:5000/health to test the server.
