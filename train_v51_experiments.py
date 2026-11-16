"""Train isolated V5.1 Top-3 feature experiments; never replace production V5."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from train_v5_daily import START_DATE, REQUIRED, download_master_files, put_object
from v5_feature_reference import FEATURES, build_features

MODEL_OBJECT = "models/v5_top3_v51_experimental.joblib"
REPORT_OBJECT = "models/v5_top3_v51_experiment_report.json"
CSV_OBJECT = "models/v5_top3_v51_experiment_comparison.csv"

# Controlled feature-group ablations, not arbitrary probability weights.
ABLATIONS = {
    "no_field_size": {"field"},
    "reduced_recent_form": {
        "last_finish", "last_finish_pct", "last3_top3", "last5_top3",
        "last10_top3", "last3_finish_pct", "last5_finish_pct", "last10_finish_pct",
    },
    "no_jockey_trainer_history": {"jockey_n", "jockey_rate", "trainer_n", "trainer_rate"},
    "no_course_going_type_history": {
        "course_n", "course_rate", "going_n", "going_rate", "type_n", "type_rate",
    },
}


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=140, max_leaf_nodes=19, learning_rate=0.055,
        l2_regularization=10, min_samples_leaf=80, random_state=42,
    )


def evaluate(name, model, columns, test):
    probabilities = model.predict_proba(test[columns].astype("float32"))[:, 1]
    target = test["top3"].astype("int8")
    scored = test[["date", "race_id", "horse", "field", "finish"]].copy()
    scored["prob_top3"] = probabilities
    eligible = scored.loc[scored.field.ge(6)].sort_values(
        ["date", "race_id", "prob_top3", "horse"],
        ascending=[True, True, False, True], kind="stable",
    )
    one_per_race = eligible.drop_duplicates(["date", "race_id"])
    picks = (one_per_race.sort_values(
        ["date", "prob_top3", "race_id"], ascending=[True, False, True], kind="stable"
    ).groupby("date", sort=False).head(10).copy())
    picks["placed"] = picks.finish.between(1, 3)
    return {
        "variant": name,
        "feature_count": len(columns),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probabilities)) if target.nunique() == 2 else None,
        "test_days": int(picks.date.nunique()),
        "daily_selections": int(len(picks)),
        "top3_selections": int(picks.placed.sum()),
        "top3_strike_rate": float(picks.placed.mean()) if len(picks) else None,
        "average_selected_probability": float(picks.prob_top3.mean()) if len(picks) else None,
    }


def run():
    started = datetime.now(timezone.utc)
    temp_dir, paths = download_master_files()
    try:
        raw = pd.concat(
            [pd.read_csv(path, usecols=REQUIRED, low_memory=False) for path in paths],
            ignore_index=True,
        )
        dates = pd.to_datetime(raw.RaceDate, format="%d-%b-%y", errors="coerce")
        if dates.isna().any():
            dates = pd.to_datetime(raw.RaceDate, dayfirst=True, errors="coerce")
        target_day = pd.Timestamp(datetime.now(ZoneInfo("Europe/London")).date())
        raw = raw.loc[dates.ge(START_DATE) & dates.lt(target_day)].copy()
        raw["RaceDate"] = dates.loc[raw.index].dt.strftime("%d-%b-%y")
        if raw.empty:
            raise RuntimeError("No master results found from 1 January 2025 onwards.")

        features = build_features(raw)
        del raw
        latest = features.loc[features.valid_finish, "date"].max()
        if pd.isna(latest):
            raise RuntimeError("No valid finishing results available.")
        test_start = max(START_DATE, latest - pd.Timedelta(days=59))
        key = ["date", "course", "time", "horse"]
        train = features.loc[features.valid_finish & features.date.lt(test_start)].drop_duplicates(key).copy()
        test = features.loc[
            features.valid_finish & features.date.ge(test_start) & features.date.le(latest)
        ].drop_duplicates(key).copy()
        if len(train) < 5000 or len(test) < 500 or train.top3.nunique() != 2 or test.top3.nunique() != 2:
            raise RuntimeError(f"Insufficient chronological holdout: train={len(train)}, test={len(test)}.")

        definitions = {"v5_baseline": list(FEATURES)}
        for name, removed in ABLATIONS.items():
            definitions[name] = [feature for feature in FEATURES if feature not in removed]

        results = []
        for name, columns in definitions.items():
            model = make_model()
            model.fit(train[columns].astype("float32"), train.top3.astype("int8"))
            metrics = evaluate(name, model, columns, test)
            results.append(metrics)
            print(json.dumps(metrics), flush=True)

        baseline = next(row for row in results if row["variant"] == "v5_baseline")
        candidates = [row for row in results if row["variant"] != "v5_baseline"]
        best = min(candidates, key=lambda row: (row["brier_score"], -(row["roc_auc"] or 0)))
        best_columns = definitions[best["variant"]]

        # Refit the shortlisted variant on all labelled history. This artifact
        # is kept separate and is not loaded by the live Flask application.
        full = features.loc[features.valid_finish].drop_duplicates(key).copy()
        full = full.loc[full.date.ge(START_DATE)]
        final_model = make_model()
        final_model.fit(full[best_columns].astype("float32"), full.top3.astype("int8"))
        bundle = {
            "model": final_model, "features": best_columns, "target": "top3",
            "model_version": "v5.1-experimental", "variant": best["variant"],
            "training_start": START_DATE.strftime("%Y-%m-%d"),
            "training_end": latest.strftime("%Y-%m-%d"),
            "training_rows": int(len(full)),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "not_for_production": True,
        }
        model_path = temp_dir / "v5_top3_v51_experimental.joblib"
        joblib.dump(bundle, model_path, compress=3)
        put_object(MODEL_OBJECT, model_path, "application/octet-stream")

        report = {
            "status": "success",
            "started_at_utc": started.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_start": START_DATE.strftime("%Y-%m-%d"),
            "holdout_start": test_start.strftime("%Y-%m-%d"),
            "holdout_end": latest.strftime("%Y-%m-%d"),
            "training_rows": int(len(train)),
            "holdout_rows": int(len(test)),
            "baseline": baseline,
            "experiments": candidates,
            "shortlisted_variant": best["variant"],
            "shortlisted_metrics": best,
            "baseline_brier_improvement": float(baseline["brier_score"] - best["brier_score"]),
            "promoted_to_production": False,
            "experimental_model_object": MODEL_OBJECT,
            "limitations": [
                "These are feature ablations, not literal percentage weights.",
                "No betting ROI is calculated because matched pre-race odds are unavailable.",
                "A chronological holdout is not proof of future profitability.",
            ],
        }
        report_path = temp_dir / "v5_top3_v51_experiment_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        put_object(REPORT_OBJECT, report_path, "application/json")
        csv_path = temp_dir / "v5_top3_v51_experiment_comparison.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)
        put_object(CSV_OBJECT, csv_path, "text/csv")
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run()
