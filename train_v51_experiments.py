"""Train isolated V5.1 Top-3 experiments; never replace production V5."""
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

# Ablations show which groups add value to the original V5 model.
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

# Disjoint specialist groups. Their probabilities are blended with the full V5
# model using explicit weights below; this is a meaningful weight experiment
# for a tree model (unlike multiplying raw input columns by constants).
FEATURE_GROUPS = {
    "recent_form": [
        "prior_runs", "last_finish", "last_finish_pct", "last3_top3", "last5_top3",
        "last10_top3", "last3_finish_pct", "last5_finish_pct", "last10_finish_pct",
    ],
    "ratings_class_distance": [
        "yards", "class_num", "weight", "orating", "age", "last_class_num",
        "last_weight", "last_orating", "last_field", "last_yards", "class_change",
        "weight_change", "or_change", "distance_change",
    ],
    "horse_connections": [
        "horse_n", "horse_rate", "jockey_n", "jockey_rate", "trainer_n", "trainer_rate",
    ],
    "race_course_conditions": [
        "field", "course_n", "course_rate", "going_n", "going_rate", "type_n",
        "type_rate", "opposition_form", "form_vs_field", "or_vs_field",
        "is_flat", "is_hurdle", "is_chase",
    ],
}

# Each profile sums to 1.0. We test balanced, recent-form-led, ratings-led,
# horse/connections-led, race-conditions-led and equal-specialist blends.
WEIGHT_PROFILES = {
    "weighted_balanced": {
        "v5_baseline": 0.35, "recent_form": 0.20, "ratings_class_distance": 0.15,
        "horse_connections": 0.10, "race_course_conditions": 0.20,
    },
    "weighted_recent_form": {
        "v5_baseline": 0.20, "recent_form": 0.40, "ratings_class_distance": 0.10,
        "horse_connections": 0.10, "race_course_conditions": 0.20,
    },
    "weighted_ratings": {
        "v5_baseline": 0.20, "recent_form": 0.10, "ratings_class_distance": 0.40,
        "horse_connections": 0.10, "race_course_conditions": 0.20,
    },
    "weighted_horse_connections": {
        "v5_baseline": 0.20, "recent_form": 0.10, "ratings_class_distance": 0.10,
        "horse_connections": 0.40, "race_course_conditions": 0.20,
    },
    "weighted_race_conditions": {
        "v5_baseline": 0.20, "recent_form": 0.10, "ratings_class_distance": 0.10,
        "horse_connections": 0.10, "race_course_conditions": 0.50,
    },
    "weighted_equal_specialists": {
        "v5_baseline": 0.20, "recent_form": 0.20, "ratings_class_distance": 0.20,
        "horse_connections": 0.20, "race_course_conditions": 0.20,
    },
}


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=140, max_leaf_nodes=19, learning_rate=0.055,
        l2_regularization=10, min_samples_leaf=80, random_state=42,
    )


def evaluate_probabilities(name, probabilities, test, feature_count):
    target = test["top3"].astype("int8")
    scored = test[["date", "race_id", "horse", "field", "finish"]].copy()
    scored["prob_top3"] = probabilities
    eligible = scored.loc[scored.field.ge(6)].sort_values(
        ["date", "race_id", "prob_top3", "horse"],
        ascending=[True, True, False, True], kind="stable",
    )
    one_per_race = eligible.drop_duplicates(["date", "race_id"])
    picks = one_per_race.sort_values(
        ["date", "prob_top3", "race_id"],
        ascending=[True, False, True], kind="stable",
    ).groupby("date", sort=False).head(10).copy()
    picks["placed"] = picks.finish.between(1, 3)
    return {
        "variant": name,
        "feature_count": feature_count,
        "brier_score": float(brier_score_loss(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(target, probabilities)) if target.nunique() == 2 else None,
        "test_days": int(picks.date.nunique()),
        "daily_selections": int(len(picks)),
        "top3_selections": int(picks.placed.sum()),
        "top3_strike_rate": float(picks.placed.mean()) if len(picks) else None,
        "average_selected_probability": float(picks.prob_top3.mean()) if len(picks) else None,
    }


def fit_probability_model(name, columns, train, test):
    model = make_model()
    model.fit(train[columns].astype("float32"), train.top3.astype("int8"))
    probabilities = model.predict_proba(test[columns].astype("float32"))[:, 1]
    return model, probabilities


def add_dp_filter(features):
    """Build Defiant Punter rules using only earlier runs and current race declarations."""
    x = features.copy()
    # Latest previous run in the same class, based only on earlier race dates.
    class_history = (
        x.loc[x.valid_finish, ["horse", "date", "class_num", "finish", "field"]]
        .drop_duplicates(["horse", "date", "class_num"])
        .sort_values(["horse", "class_num", "date"], kind="stable")
    )
    class_groups = class_history.groupby(["horse", "class_num"], sort=False)
    class_history["dp_same_class_last_finish"] = class_groups["finish"].shift(1)
    class_history["dp_same_class_last_field"] = class_groups["field"].shift(1)
    class_history = class_history[[
        "horse", "date", "class_num", "dp_same_class_last_finish", "dp_same_class_last_field"
    ]]
    x = x.merge(class_history, on=["horse", "date", "class_num"], how="left", sort=False)

    # Has this horse previously raced over today's exact distance?
    distance_history = (
        x[["horse", "date", "yards"]].drop_duplicates(["horse", "date", "yards"])
        .sort_values(["horse", "yards", "date"], kind="stable")
    )
    distance_groups = distance_history.groupby(["horse", "yards"], sort=False)
    distance_history["dp_prior_distance_runs"] = distance_groups.cumcount()
    x = x.merge(
        distance_history[["horse", "date", "yards", "dp_prior_distance_runs"]],
        on=["horse", "date", "yards"], how="left", sort=False,
    )

    # Rank lower carried weights and older ages within each race.
    x["dp_weight_percentile"] = x.groupby("race_id")["weight"].rank(
        method="average", ascending=True, pct=True
    )
    x["dp_age_rank_oldest"] = x.groupby("race_id")["age"].rank(
        method="min", ascending=False
    )
    x["dp_same_class_top20"] = (
        x["dp_same_class_last_finish"].notna()
        & x["dp_same_class_last_field"].gt(0)
        & (x["dp_same_class_last_finish"] <=
           (0.20 * x["dp_same_class_last_field"].fillna(0)).apply(__import__("math").ceil))
    )
    x["dp_filter"] = (
        x["class_change"].gt(0)  # class number rises: easier class in this dataset
        & x["dp_same_class_top20"]
        & x["dp_weight_percentile"].le(0.50)
        & x["dp_age_rank_oldest"].gt(3)
        & x["dp_prior_distance_runs"].gt(0)
    )
    return x


def evaluate_dp_filter(test, baseline_probabilities):
    scored = test[["date", "race_id", "horse", "field", "finish", "dp_filter"]].copy()
    scored["prob_top3"] = baseline_probabilities
    eligible = scored.loc[scored.dp_filter & scored.field.ge(6)].sort_values(
        ["date", "prob_top3", "race_id", "horse"],
        ascending=[True, False, True, True], kind="stable",
    )
    picks = eligible.groupby("date", sort=False).head(10).copy()
    picks["placed"] = picks.finish.between(1, 3)
    days = int(picks.date.nunique())
    return {
        "variant": "option_11_dp_filter",
        "filter_definition": {
            "class_drop": "Today's class number is higher (easier) than the horse's previous run.",
            "same_class_form": "Most recent previous run in today's class finished in the top 20% of that field.",
            "weight": "Today's carried weight is in the lowest 50% of runners in the race.",
            "age": "Horse is not among the three oldest runners in the race.",
            "distance": "Horse has previously run today's exact distance.",
        },
        "eligible_runner_count": int(scored.dp_filter.sum()),
        "test_days": days,
        "daily_selections": int(len(picks)),
        "top3_selections": int(picks.placed.sum()),
        "top3_strike_rate": float(picks.placed.mean()) if len(picks) else None,
        "average_selected_probability": float(picks.prob_top3.mean()) if len(picks) else None,
        "selection_rule": "Among DP-filter qualifiers, select highest unchanged V5 baseline probability, up to 10 per day.",
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

        features = add_dp_filter(build_features(raw))
        del raw
        latest = features.loc[features.valid_finish, "date"].max()
        if pd.isna(latest):
            raise RuntimeError("No valid finishing results available.")
        test_start = max(START_DATE, latest - pd.Timedelta(days=59))
        key = ["date", "course", "time", "horse"]
        train = features.loc[
            features.valid_finish & features.date.lt(test_start)
        ].drop_duplicates(key).copy()
        test = features.loc[
            features.valid_finish & features.date.ge(test_start) & features.date.le(latest)
        ].drop_duplicates(key).copy()
        if len(train) < 5000 or len(test) < 500 or train.top3.nunique() != 2 or test.top3.nunique() != 2:
            raise RuntimeError(f"Insufficient chronological holdout: train={len(train)}, test={len(test)}.")

        # Guard against accidentally omitting or repeating any V5 input feature.
        grouped = [feature for group in FEATURE_GROUPS.values() for feature in group]
        if len(grouped) != len(set(grouped)) or set(grouped) != set(FEATURES):
            missing = sorted(set(FEATURES) - set(grouped))
            extra = sorted(set(grouped) - set(FEATURES))
            raise RuntimeError(f"Feature groups must partition FEATURES exactly. Missing={missing}; extra={extra}")
        for name, weights in WEIGHT_PROFILES.items():
            if set(weights) != {"v5_baseline", *FEATURE_GROUPS.keys()} or abs(sum(weights.values()) - 1.0) > 1e-9:
                raise RuntimeError(f"Invalid weights in profile {name}.")

        results = []
        definitions = {"v5_baseline": list(FEATURES)}
        for name, removed in ABLATIONS.items():
            definitions[name] = [feature for feature in FEATURES if feature not in removed]

        # First compare the unchanged baseline and controlled ablations.
        component_probabilities = {}
        for name, columns in definitions.items():
            _, probabilities = fit_probability_model(name, columns, train, test)
            if name == "v5_baseline":
                component_probabilities["v5_baseline"] = probabilities
            metrics = evaluate_probabilities(name, probabilities, test, len(columns))
            results.append(metrics)
            print(json.dumps(metrics), flush=True)

        # Train one specialist model per independent filter group, then test
        # several explicit blends against the full-feature V5 baseline.
        for group_name, columns in FEATURE_GROUPS.items():
            _, probabilities = fit_probability_model(group_name, columns, train, test)
            component_probabilities[group_name] = probabilities
        weight_results = []
        for profile_name, weights in WEIGHT_PROFILES.items():
            combined = sum(
                component_probabilities[component] * weight
                for component, weight in weights.items()
            )
            metrics = evaluate_probabilities(
                profile_name, combined, test,
                sum(len(FEATURE_GROUPS[k]) for k in FEATURE_GROUPS) + len(FEATURES),
            )
            metrics["weights"] = weights
            results.append(metrics)
            weight_results.append(metrics)
            print(json.dumps(metrics), flush=True)

        baseline = next(row for row in results if row["variant"] == "v5_baseline")
        dp_filter_result = evaluate_dp_filter(test, component_probabilities["v5_baseline"])
        results.append(dp_filter_result)
        print(json.dumps(dp_filter_result), flush=True)
        candidates = [
            row for row in results
            if row["variant"] not in {"v5_baseline", "option_11_dp_filter"}
        ]
        best = min(candidates, key=lambda row: (row["brier_score"], -(row["roc_auc"] or 0)))
        best_columns = definitions.get(best["variant"])

        # Refit the winning experimental variant on all labelled history.
        full = features.loc[features.valid_finish].drop_duplicates(key).copy()
        full = full.loc[full.date.ge(START_DATE)]
        if best_columns is not None:
            final_model = make_model()
            final_model.fit(full[best_columns].astype("float32"), full.top3.astype("int8"))
            bundle = {
                "model": final_model, "features": best_columns, "target": "top3",
                "model_version": "v5.1-experimental", "variant": best["variant"],
            }
        else:
            profile_weights = WEIGHT_PROFILES[best["variant"]]
            ensemble_models = {}
            for group_name, columns in FEATURE_GROUPS.items():
                specialist = make_model()
                specialist.fit(full[columns].astype("float32"), full.top3.astype("int8"))
                ensemble_models[group_name] = {"model": specialist, "features": columns}
            baseline_model = make_model()
            baseline_model.fit(full[FEATURES].astype("float32"), full.top3.astype("int8"))
            bundle = {
                "model": baseline_model, "features": list(FEATURES),
                "ensemble_models": ensemble_models, "weights": profile_weights,
                "target": "top3", "model_version": "v5.1-weighted-experimental",
                "variant": best["variant"],
            }
        bundle.update({
            "training_start": START_DATE.strftime("%Y-%m-%d"),
            "training_end": latest.strftime("%Y-%m-%d"),
            "training_rows": int(len(full)),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "not_for_production": True,
        })
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
            "ablation_experiments": [
                row for row in results
                if row["variant"] in ABLATIONS
            ],
            "weight_profiles": weight_results,
            "option_11_dp_filter": dp_filter_result,
            "shortlisted_variant": best["variant"],
            "shortlisted_metrics": best,
            "baseline_brier_improvement": float(baseline["brier_score"] - best["brier_score"]),
            "promoted_to_production": False,
            "experimental_model_object": MODEL_OBJECT,
            "limitations": [
                "The DP filter is a deterministic rule-based shortlist, not a trained probability model. Its strike rate and selection volume are reported separately.",
                "Weight profiles blend separately trained specialist probabilities with the full V5 baseline.",
                "The same chronological holdout is used to compare and shortlist variants; repeat testing on a later untouched period is required before promotion.",
                "No betting ROI is calculated because matched pre-race odds are unavailable.",
                "A higher Top-3 strike rate alone does not establish profitability.",
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
