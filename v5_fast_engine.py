"""Memory-bounded V5 inference; uses the original frozen feature builder.

Only historical rows for today's horses enter build_features. Jockey/trainer
rates are computed from ALL historical runners in streaming chunks to retain
exact historical definitions. No model training or feature changes.
"""
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from v5_engine import yards_from_distance, TYPE_MAP, select_daily
from v5_feature_reference import build_features, clean_text

NEEDED=['RaceDate','Course','RaceTime','HorseName','FPos','Ran','Type','Going','Yards','Class','WeightLBS','OR','Jockey','Trainer','Age']

def predict_v5_fast(paths, runners, model_path, race_date, chunksize=12000, target='top3'):
    day=pd.Timestamp(race_date)
    r=runners.copy().reset_index(drop=True)
    if r.empty:raise ValueError('No racecard runners')
    required=['Horse','Course','Time','Final_Race_Type','Final_Going','RFB_Age','RFB_WeightLBS','RFB_OR','Final_Class']
    missing=[c for c in required if c not in r]
    if missing:raise ValueError('Racecard missing: '+', '.join(missing))
    if r[['Horse','Course','Time']].isna().any().any():raise ValueError('Missing horse/course/time')
    if r.duplicated(['Course','Time','Horse']).any():raise ValueError('Duplicate horse within race')
    for c in ('Jockey','Trainer','Distance'):
        if c not in r:r[c]=''
    r['field']=r.groupby(['Course','Time'])['Horse'].transform('size')
    if 'Yards' not in r:r['Yards']=r.Distance.map(yards_from_distance)
    else:r['Yards']=pd.to_numeric(r.Yards,errors='coerce').fillna(r.Distance.map(yards_from_distance))
    typ=r.Final_Race_Type.astype('string').str.strip().str.lower().map(TYPE_MAP)
    if typ.isna().any():raise ValueError('Unrecognised race type in runner table')
    today=pd.DataFrame({'RaceDate':day.strftime('%d-%b-%y'),'Course':r.Course,'RaceTime':r.Time,
        'HorseName':r.Horse,'FPos':np.nan,'Ran':r.field,'Type':typ,'Going':r.Final_Going,
        'Yards':r.Yards,'Class':pd.to_numeric(r.Final_Class.astype(str).str.extract(r'(\d+)')[0],errors='coerce'),
        'WeightLBS':r.RFB_WeightLBS,'OR':r.RFB_OR,'Jockey':r.Jockey,'Trainer':r.Trainer,'Age':r.RFB_Age})
    horses=set(clean_text(today.HorseName).str.replace(r'\s*\((ire|gb|fr|usa|ger|aus|nz)\)$','',regex=True))
    jockeys=set(clean_text(today.Jockey))
    trainers=set(clean_text(today.Trainer))
    jockey_stats=defaultdict(lambda:[0,0]);trainer_stats=defaultdict(lambda:[0,0])
    selected=[]
    for path in paths:
        path=Path(path)
        if not path.is_file():raise FileNotFoundError(f'Master file unavailable: {path.name}')
        cols=pd.read_csv(path,nrows=0).columns
        missing=[c for c in NEEDED if c not in cols]
        if missing:raise ValueError(f'{path.name} missing: '+', '.join(missing))
        for chunk in pd.read_csv(path,usecols=NEEDED,chunksize=chunksize,low_memory=False):
            dates=pd.to_datetime(chunk.RaceDate,format='%d-%b-%y',errors='coerce')
            chunk=chunk.loc[dates.lt(day)].copy()
            if chunk.empty:continue
            # The original builder counts every historical row for jockey/trainer,
            # including non-runners (top3=0), and excludes invalid horse/course rows.
            normhorse=clean_text(chunk.HorseName).str.replace(r'\s*\((ire|gb|fr|usa|ger|aus|nz)\)$','',regex=True)
            valid=normhorse.ne('') & clean_text(chunk.Course).ne('')
            chunk=chunk.loc[valid].copy();normhorse=normhorse.loc[valid]
            if chunk.empty:continue
            wins=pd.to_numeric(chunk.FPos,errors='coerce').between(1,3).to_numpy(dtype=np.int64)
            for col, wanted, stats in [('Jockey',jockeys,jockey_stats),('Trainer',trainers,trainer_stats)]:
                normalized=clean_text(chunk[col]);mask=normalized.isin(wanted)
                if mask.any():
                    z=pd.DataFrame({'key':normalized.loc[mask].to_numpy(),'win':wins[mask.to_numpy()]})
                    counts=z.groupby('key',sort=False)['win'].agg(['sum','count'])
                    for key,row in counts.iterrows():
                        stats[key][0]+=int(row['sum']);stats[key][1]+=int(row['count'])
            mask=normhorse.isin(horses)
            if mask.any():selected.append(chunk.loc[mask,NEEDED])
    history=pd.concat(selected,ignore_index=True) if selected else pd.DataFrame(columns=NEEDED)
    both=pd.concat([history,today[NEEDED]],ignore_index=True)
    del selected,history
    feats=build_features(both)
    current=feats.loc[feats.date.eq(day)].copy()
    if len(current)!=len(today):raise ValueError('Some racecard runners were discarded by feature validation')
    # Replace rates computed from the horse-only subset with rates across the
    # COMPLETE historical dataset, exactly as in build_features(full_history).
    for prefix,stats in [('jockey',jockey_stats),('trainer',trainer_stats)]:
        values=current[prefix].map(lambda k:stats[k])
        current[prefix+'_n']=values.map(lambda a:a[1]).astype('float32')
        current[prefix+'_rate']=values.map(lambda a:(a[0]+1)/(a[1]+3)).astype('float32')
    pack=joblib.load(model_path) # Trusted model bundled with this repository.
    missing=set(pack['features'])-set(current.columns)
    if missing:raise ValueError('Missing model features: '+str(sorted(missing)))
    if pack.get('target', 'top3') != target:
        raise ValueError('Model target mismatch: expected ' + target + ', got ' + str(pack.get('target')))
    probability_column = 'prob_win' if target == 'win' else 'prob_top3'
    label = 'Win Probability' if target == 'win' else 'Top 3 Probability'
    current[probability_column]=pack['model'].predict_proba(current[pack['features']].astype('float32'))[:,1]
    out=current[['course','time','horse','field',probability_column,'prior_runs']].copy()
    out.columns=['Course','Time','Horse','Runners',label,'Prior Runs']
    out=out[out.Runners>=6].copy()
    out['Race']=out.Course.astype(str)+' '+out.Time.astype(str)
    out['Race Rank']=out.groupby('Race')['Top 3 Probability'].rank(ascending=False,method='first').astype(int)
    return out.sort_values('Top 3 Probability',ascending=False).reset_index(drop=True)
