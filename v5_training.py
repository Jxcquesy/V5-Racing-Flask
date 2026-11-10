"""Daily V5 Top-3 retraining using all completed results from 1 Jan 2025 to the day before the target."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from v5_feature_reference import build_features, FEATURES

START = pd.Timestamp("2025-01-01")
NEEDED = ['RaceDate','Course','RaceTime','HorseName','FPos','Ran','Type','Going',
          'Yards','Class','WeightLBS','OR','Jockey','Trainer','Age']
KEY = ['date','course','time','horse']

def train_daily_model(paths, model_path, target_day, source_signature):
    """Always fit a fresh model for target_day; only completed pre-target results are eligible."""
    target = pd.Timestamp(target_day).normalize()
    frames = []
    for path in paths:
        path = Path(path)
        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in NEEDED if c not in header]
        if missing:
            raise ValueError(f"{path.name} missing training columns: {', '.join(missing)}")
        frame = pd.read_csv(path, usecols=NEEDED, low_memory=False)
        dates = pd.to_datetime(frame['RaceDate'], format='%d-%b-%y', errors='coerce')
        keep = dates.ge(START) & dates.lt(target)
        frame = frame.loc[keep].copy()
        if not frame.empty:
            frames.append(frame)
        del frame, dates

    if not frames:
        raise ValueError("No completed race results found between 1 January 2025 and the day before the prediction date.")

    raw = pd.concat(frames, ignore_index=True)
    del frames
    # The identity key prevents accidental duplicate runners, while retaining repeat
    # appearances by the same horse in different races.
    raw_dates = pd.to_datetime(raw['RaceDate'], format='%d-%b-%y', errors='coerce')
    raw['__date'] = raw_dates
    raw['__course'] = raw['Course'].astype('string').fillna('').str.strip().str.casefold()
    raw['__time'] = raw['RaceTime'].astype('string').fillna('').str.strip()
    raw['__horse'] = raw['HorseName'].astype('string').fillna('').str.strip().str.casefold()
    raw = raw.drop_duplicates(['__date','__course','__time','__horse'], keep='first')
    raw = raw.drop(columns=['__date','__course','__time','__horse'])
    if raw.empty:
        raise ValueError("No usable historical rows remain after duplicate removal.")

    features = build_features(raw)
    train = features.loc[features['valid_finish']].copy()
    train = train.drop_duplicates(KEY, keep='first')
    if len(train) < 100:
        raise ValueError(f"Only {len(train):,} valid training runners are available; need at least 100.")
    if train['top3'].nunique() < 2:
        raise ValueError("Training data contains only one outcome class; cannot fit the Top 3 model.")

    X = train[FEATURES].astype('float32').replace([np.inf, -np.inf], np.nan)
    y = train['top3'].astype('int8')
    model = HistGradientBoostingClassifier(
        max_iter=140, max_leaf_nodes=19, learning_rate=.055,
        l2_regularization=10, min_samples_leaf=80, random_state=42
    )
    model.fit(X, y)

    trained_until = train['date'].max().strftime('%Y-%m-%d')
    pack = {
        'model': model,
        'features': FEATURES,
        'target': 'top3',
        'training_start': train['date'].min().strftime('%Y-%m-%d'),
        'requested_training_start': START.strftime('%Y-%m-%d'),
        'trained_until': trained_until,
        'training_day': target.strftime('%Y-%m-%d'),
        'training_rows': int(len(train)),
        'training_positive_rows': int(y.sum()),
        'source_signature': source_signature,
        'training_method': 'daily_expanding_window'
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_name(model_path.name + '.tmp')
    try:
        joblib.dump(pack, temp_path, compress=3)
        os.replace(temp_path, model_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return {k: pack[k] for k in ('training_start','trained_until','training_day','training_rows','training_positive_rows','training_method')}
