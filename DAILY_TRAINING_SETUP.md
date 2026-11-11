# Daily V5 training setup

The daily trainer runs on GitHub Actions, not on the Render web service. This prevents model fitting from consuming the web process memory and taking the website offline.

## One-time setup

In GitHub, open **Jxcquesy/V5-Racing-Flask → Settings → Secrets and variables → Actions** and add these repository secrets:

- `SUPABASE_URL`: the same Supabase project URL already configured on Render.
- `SUPABASE_SERVICE_ROLE_KEY`: the same Supabase service-role key already configured on Render.

Do not paste either value into chat, source code, or a committed file. The existing private `v5-master` bucket must contain yearly files named `V5_Master_2025.csv`, `V5_Master_2026.csv`, and so on.

The workflow writes two objects into that same bucket:

- `models/v5_top3_model.joblib`: the existing Top 3 model (same target and feature definitions).
- `models/v5_winner_model.joblib`: the separate first-place / WIN probability model.
- `models/v5_winner_backtest.json`: chronological 60-day holdout metrics.
- `models/v5_winner_backtest.csv`: every daily Top 10 selection and its result.
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

- Keeps the existing V5 Top 3 model and trains it as before.
- Also trains a separate Winner model against finishing position 1, using the same 42 V5 features and HistGradientBoosting model family.
- Runs a chronological holdout backtest: train before the final 60 calendar days, then select up to 10 horses per test day, with no more than one selection per race and a minimum field size of 6.
- Uses valid completed results dated from 1 January 2025 up to, but not including, the current UK race date.
- If the latest results have not been uploaded, it uses the latest available date in Supabase; it does not wait for yesterday's results.
- Retrains each day regardless of the previous model's betting or prediction performance.
- Checks both models' feature compatibility and probability output before publication.
- Reports winner strike rate, field-size random baseline, winner capture rate, Brier score and ROC AUC.
- Does not report profit/ROI because the Master results CSV does not contain matched pre-race bookmaker odds; odds must not be invented.
- The old model remains available in the Render app if training fails or no new model has been published.

The training job does not fetch new race results. Results must first be added to the yearly Master CSVs in Supabase.

## Safety

Do not change the Render start command or add training to a Flask route. Do not make the Supabase bucket public. The service-role key must remain a GitHub Actions secret and a Render environment variable only.
