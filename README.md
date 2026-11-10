# V5 Racing Predictor — Flask / Render

This is a **separate Flask rebuild** of the uploaded V5 Streamlit repository. The V5 Flask application uses the existing feature pipeline and PDF parser. V4 is not involved. The bundled model is retained as a fallback; a fresh model is fitted for each new prediction date.

## Deploy to Render

1. Create a **new GitHub repository** for these files (do not overwrite V4 or the current V5 Streamlit repo). Upload the contents of this folder to its root.
2. In Render choose **New → Blueprint** and connect the new repository. `render.yaml` configures a Python web service. Or create a Web Service manually with `pip install -r requirements.txt` and `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app`.
3. In Render **Environment** set `APP_PASSWORD` to a strong private login password, `SUPABASE_URL` to your existing Supabase project URL, and `SUPABASE_SERVICE_ROLE_KEY` to your existing service-role key. `SECRET_KEY` is generated automatically by the Blueprint; for manual deployments add a long random secret. **Never commit these values to GitHub.**
4. Set Python 3.12 where Render allows selecting it. Log in with `APP_PASSWORD`, open History, and confirm the old records are visible.
5. Upload your existing historical Master CSV on Data & Training. It was **not included** in the supplied ZIP. Upload a corrected racecard CSV or PDFs to run predictions.

## Important limitations

- **Master CSV storage is temporary** on Render's free instance filesystem. It can be lost after restart/redeploy. Download a backup after every merge. Permanent master storage needs a database or object storage integration.
- **Daily retraining is enabled when selections are generated for a new race date.** Training uses completed results dated 1 January 2025 through the latest date before the target race day. If the latest results have not been uploaded, V5 trains on the latest available data already in Supabase. It retrains again if the stored yearly Master files change.
- If a training run fails, the last successfully saved model is kept and used for predictions; the dashboard reports the last successful training date. Training runs on demand before predictions, not as a separate midnight background job.
- Automatic results fetching is not implemented: results still need to be merged/uploaded to Supabase.
- The PDF parser is best-effort and is reused unchanged. Unlike the Streamlit app, this Flask version currently has **no editable racecard verification grid**; upload a corrected CSV if the PDF extraction needs corrections.
- Supabase betting history uses the existing `v5_bets` table and compatible fields. No migration or history re-entry is needed. The £/unit setting is used for display but is not stored because the supplied Supabase schema does not include a unit-value column.
- The app uses password protection and server-side Supabase credentials. The login password should be private; for multi-user access add proper account-based authentication.
- The model is loaded from a trusted, owner-supplied joblib file. Do not replace it with an untrusted file.
- Render free web services may sleep when idle, and a paid plan may be required for more reliable availability.

## Local test

```bash
python -m venv .venv
# activate the virtual environment
pip install -r requirements.txt
# set SECRET_KEY, APP_PASSWORD, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in your shell
python app.py
```

Visit http://127.0.0.1:5000/health to test the server.
