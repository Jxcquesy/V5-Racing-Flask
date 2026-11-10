# Daily V5 training setup

The daily trainer runs on GitHub Actions, not on the Render web service. This prevents model fitting from consuming the web process memory and taking the website offline.

## One-time setup

In GitHub, open **Jxcquesy/V5-Racing-Flask → Settings → Secrets and variables → Actions** and add these repository secrets:

- `SUPABASE_URL`: the same Supabase project URL already configured on Render.
- `SUPABASE_SERVICE_ROLE_KEY`: the same Supabase service-role key already configured on Render.

Do not paste either value into chat, source code, or a committed file. The existing private `v5-master` bucket must contain yearly files named `V5_Master_2025.csv`, `V5_Master_2026.csv`, and so on.

The workflow writes two objects into that same bucket:

- `models/v5_top3_model.joblib`: the latest successfully trained model.
- `models/training_status.json`: the last training outcome and date coverage.

No new bucket is required.

## First test

1. Open the repository's **Actions** tab.
2. Select **Daily V5 model training**.
3. Select **Run workflow** on the `main` branch.
4. Open the run and inspect the **Train and publish V5** log.
5. A successful run uploads the model and status. The V5 Data page then displays the training date range and runner count.

The workflow is also scheduled daily at 07:00 UTC. GitHub Actions may delay scheduled runs during busy periods, so use **Run workflow** for an immediate manual run.

## Training rules

- Uses the existing V5 feature definitions and HistGradientBoosting model family.
- Uses valid completed results dated from 1 January 2025 up to, but not including, the current UK race date.
- If the latest results have not been uploaded, it uses the latest available date in Supabase; it does not wait for yesterday's results.
- Retrains each day regardless of the previous model's betting or prediction performance.
- Checks the new model's feature compatibility and probability output before publication.
- The old model remains available in the Render app if training fails or no new model has been published.

The training job does not fetch new race results. Results must first be added to the yearly Master CSVs in Supabase.

## Safety

Do not change the Render start command or add training to a Flask route. Do not make the Supabase bucket public. The service-role key must remain a GitHub Actions secret and a Render environment variable only.
