"""Daily V5 retraining, isolated from the Render web process.

Reads yearly master CSVs from private Supabase Storage, trains the existing
V5 feature/model family on results dated 2025-01-01 through the latest
available result date, then atomically publishes the model to Storage.
"""
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier

from v5_feature_reference import FEATURES, build_features

BUCKET = os.getenv("MASTER_BUCKET", "v5-master")
START_DATE = pd.Timestamp("2025-01-01")
MODEL_OBJECT = "models/v5_top3_model.joblib"
STATUS_OBJECT = "models/training_status.json"
REQUIRED = [
    "RaceDate", "Course", "RaceTime", "HorseName", "FPos", "Ran", "Type",
    "Going", "Yards", "Class", "WeightLBS", "OR", "Jockey", "Trainer", "Age",
]


def config():
    base = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return base, key


def headers(content_type=None):
    _, key = config()
    out = {"apikey": key, "Authorization": "Bearer " + key}
    if content_type:
        out["Content-Type"] = content_type
    return out


def object_url(name):
    base, _ = config()
    return f"{base}/storage/v1/object/{quote(BUCKET)}/{quote(name, safe='/')}"


def put_object(name, path, content_type):
    h = headers(content_type)
    h["x-upsert"] = "true"
    with open(path, "rb") as fh:
        response = requests.post(object_url(name), headers=h, data=fh, timeout=300)
    response.raise_for_status()


def put_status(payload):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
        json.dump(payload, f, indent=2)
        temp = f.name
    try:
        put_object(STATUS_OBJECT, temp, "application/json")
    finally:
        Path(temp).unlink(missing_ok=True)


def download_master_files():
    base, _ = config()
    response = requests.post(
        f"{base}/storage/v1/object/list/{BUCKET}",
        headers=headers("application/json"),
        json={"prefix": "", "limit": 1000, "offset": 0},
        timeout=60,
    )
    response.raise_for_status()
    years = []
    for item in response.json():
        name = item.get("name", "")
        if name.startswith("V5_Master_") and name.endswith(".csv"):
            try:
                year = int(name.removeprefix("V5_Master_").removesuffix(".csv"))
            except ValueError:
                continue
            if year >= 2025:
                years.append((year, name))
    years.sort()
    if not years or years[0][0] > 2025:
        raise RuntimeError("V5_Master_2025.csv is missing from the Supabase master bucket.")

    temp_dir = Path(tempfile.mkdtemp(prefix="v5-training-"))
    paths = []
    for year, name in years:
        target = temp_dir / name
        with requests.get(object_url(name), headers=headers(), stream=True, timeout=300) as response:
            response.raise_for_status()
            with target.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        out.write(chunk)
        paths.append(target)
        print(f"Downloaded {name}: {target.stat().st_size:,} bytes", flush=True)
    return temp_dir, paths


def run_training():
    started = datetime.now(timezone.utc)
    temp_dir, paths = download_master_files()
    try:
        frames = []
        for path in paths:
            frame = pd.read_csv(path, usecols=REQUIRED, low_memory=False)
            frames.append(frame)
            print(f"Read {path.name}: {len(frame):,} rows", flush=True)
        raw = pd.concat(frames, ignore_index=True)
        del frames

        dates = pd.to_datetime(raw["RaceDate"], format="%d-%b-%y", errors="coerce")
        if dates.isna().any():
            dates = pd.to_datetime(raw["RaceDate"], dayfirst=True, errors="coerce")
        target_day = pd.Timestamp(datetime.now(ZoneInfo("Europe/London")).date())
        raw = raw.loc[dates.ge(START_DATE) & dates.lt(target_day)].copy()
        raw["RaceDate"] = dates.loc[raw.index].dt.strftime("%d-%b-%y")
        if raw.empty:
            raise RuntimeError("No master results found from 1 January 2025 onwards.")

        latest_date = pd.to_datetime(raw["RaceDate"], format="%d-%b-%y").max()
        print(
            f"Training data: {len(raw):,} runners; {START_DATE.date()} to {latest_date.date()}",
            flush=True,
        )

        features = build_features(raw)
        del raw
        train = features.loc[features["valid_finish"]].copy()
        train = train.drop_duplicates(["date", "course", "time", "horse"])
        train = train.loc[train["date"].ge(START_DATE)]
        if len(train) < 5000:
            raise RuntimeError(f"Only {len(train):,} valid training rows; refusing to publish a weak/incomplete run.")
        if train["top3"].nunique() != 2:
            raise RuntimeError("Training data does not contain both Top-3 and non-Top-3 examples.")

        x = train[FEATURES].astype("float32")
        y = train["top3"].astype("int8")
        model = HistGradientBoostingClassifier(
            max_iter=140,
            max_leaf_nodes=19,
            learning_rate=0.055,
            l2_regularization=10,
            min_samples_leaf=80,
            random_state=42,
        )
        model.fit(x, y)

        # Technical integrity check only; no performance threshold can block
        # retraining, as requested. Confirm the model returns valid probabilities.
        sample = x.iloc[: min(256, len(x))]
        probabilities = model.predict_proba(sample)[:, 1]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("New model failed its probability integrity check.")

        bundle = {
            "model": model,
            "features": FEATURES,
            "cutoff": latest_date.strftime("%Y-%m-%d"),
            "training_start": START_DATE.strftime("%Y-%m-%d"),
            "training_end": latest_date.strftime("%Y-%m-%d"),
            "prediction_day": target_day.strftime("%Y-%m-%d"),
            "training_rows": int(len(train)),
            "target": "top3",
            "model_version": "daily-v1",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        model_path = temp_dir / "v5_top3_model.joblib"
        joblib.dump(bundle, model_path, compress=3)
        # Publish the model first; status=success is the commit marker.
        put_object(MODEL_OBJECT, model_path, "application/octet-stream")

        finished = datetime.now(timezone.utc)
        status = {
            "status": "success",
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "training_start": START_DATE.strftime("%Y-%m-%d"),
            "training_end": latest_date.strftime("%Y-%m-%d"),
            "training_rows": int(len(train)),
            "source_years": [p.stem.replace("V5_Master_", "") for p in paths],
            "message": "New V5 model successfully trained and published.",
        }
        put_status(status)
        print(json.dumps(status, indent=2), flush=True)
        return status
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        run_training()
    except Exception as exc:
        failure = {
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "message": str(exc)[:1000],
        }
        try:
            put_status(failure)
        except Exception:
            print("Could not publish training failure status:", traceback.format_exc(), file=sys.stderr)
        raise
