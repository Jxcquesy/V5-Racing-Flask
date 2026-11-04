"""Frozen V5 inference. Historical form is calculated only from dates before the target day."""
import io, re
import joblib
import numpy as np
import pandas as pd
from v5_feature_reference import build_features

TYPE_MAP={'flat':'f','all weather':'a','hurdle':'h','chase':'c','nh flat':'b','bumper':'b'}

def yards_from_distance(value):
    s=str(value).lower().replace('½','.5').replace('¼','.25')
    miles=re.search(r'(\d+(?:\.\d+)?)\s*m(?!in)',s)
    furlongs=re.search(r'(\d+(?:\.\d+)?)\s*f',s)
    yards=re.search(r'(\d+)\s*y',s)
    if not any((miles,furlongs,yards)):return np.nan
    return (float(miles.group(1))*1760 if miles else 0)+(float(furlongs.group(1))*220 if furlongs else 0)+(int(yards.group(1)) if yards else 0)

def predict_v5(history_bytes,runners,model_bytes,race_date):
    history=pd.read_csv(io.BytesIO(history_bytes),low_memory=False)
    required=['RaceDate','Course','RaceTime','HorseName','FPos','Ran','Type','Going','Yards','Class','WeightLBS','OR','Jockey','Trainer','Age']
    missing=[c for c in required if c not in history]
    if missing:raise ValueError('Master history missing: '+', '.join(missing))
    day=pd.Timestamp(race_date)
    hdates=pd.to_datetime(history.RaceDate,format='%d-%b-%y',errors='coerce')
    history=history.loc[hdates.lt(day)].copy()
    r=runners.copy().reset_index(drop=True)
    if r.empty:raise ValueError('No racecard runners')
    required_runner=['Horse','Course','Time','Final_Race_Type','Final_Going','RFB_Age','RFB_WeightLBS','RFB_OR','Final_Class']
    missing=[c for c in required_runner if c not in r]
    if missing:raise ValueError('Racecard missing: '+', '.join(missing))
    if r[['Horse','Course','Time']].isna().any().any():raise ValueError('Missing horse/course/time')
    if r.duplicated(['Course','Time','Horse']).any():raise ValueError('Duplicate horse within race')
    r['field']=r.groupby(['Course','Time'])['Horse'].transform('size')
    if 'Distance' not in r:r['Distance']=''
    if 'Yards' not in r:r['Yards']=r.Distance.map(yards_from_distance)
    else:r['Yards']=pd.to_numeric(r.Yards,errors='coerce').fillna(r.Distance.map(yards_from_distance))
    typ=r.Final_Race_Type.astype('string').str.strip().str.lower().map(TYPE_MAP)
    # Fail closed for unsupported race types: otherwise the model would silently treat jumps as flat.
    if typ.isna().any():raise ValueError('Unrecognised race type in runner table; correct before running V5')
    today=pd.DataFrame({
        'RaceDate':day.strftime('%d-%b-%y'),'Course':r.Course,'RaceTime':r.Time,'HorseName':r.Horse,
        'FPos':np.nan,'Ran':r.field,'Type':typ,'Going':r.Final_Going,
        'Yards':r.Yards,'Class':pd.to_numeric(r.Final_Class.astype(str).str.extract(r'(\d+)')[0],errors='coerce'),
        'WeightLBS':r.RFB_WeightLBS,'OR':r.RFB_OR,
        'Jockey':r.Jockey if 'Jockey' in r else '',
        'Trainer':r.Trainer if 'Trainer' in r else '',
        'Age':r.RFB_Age})
    # Distinct synthetic date/horse keys; no past result or target information is supplied for today.
    both=pd.concat([history[required],today[required]],ignore_index=True)
    feats=build_features(both)
    current=feats[feats.date.eq(day)].copy()
    pack=joblib.load(io.BytesIO(model_bytes)) # Load only a trusted owner-provided model.
    missing=set(pack['features'])-set(current.columns)
    if missing:raise ValueError('Missing model features: '+str(sorted(missing)))
    current['prob_top3']=pack['model'].predict_proba(current[pack['features']].astype('float32'))[:,1]
    out=current[['course','time','horse','field','prob_top3','prior_runs']].copy()
    out.columns=['Course','Time','Horse','Runners','Top 3 Probability','Prior Runs']
    out=out[out.Runners>=6].copy()
    out['Race']=out.Course.astype(str)+' '+out.Time.astype(str)
    out['Race Rank']=out.groupby('Race')['Top 3 Probability'].rank(ascending=False,method='first').astype(int)
    out=out.sort_values('Top 3 Probability',ascending=False).reset_index(drop=True)
    return out

def select_daily(predictions,n=10):
    # The highest-scoring runner in each distinct race; no duplicated race in treble.
    return predictions.drop_duplicates('Race').head(n).copy()
