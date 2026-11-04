"""V5: chronological horse racing Top-3 model and 27-day audit.
Run: python v5_train_backtest.py --csv path/to/master.csv --out v5_results
Requires: pandas, numpy, scikit-learn, joblib.
No odds are invented. Frozen model trained on races before 2026-09-01.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

CUT = pd.Timestamp('2026-09-01')
END = pd.Timestamp('2026-09-28')

def clean_text(s):
    return s.astype('string').fillna('').str.strip().str.lower().str.replace(r'\s+',' ',regex=True)

def norm_going(s):
    v=clean_text(s)
    out=np.select([v.str.contains('heavy'),v.str.contains('soft'),v.str.contains('firm'),v.str.contains('good'),v.str.contains('standard'),v.str.contains('slow'),v.str.contains('fast')],['heavy','soft','firm','good','standard','slow','fast'],default='unknown')
    return pd.Series(out,index=s.index)

def historical_rate(df,key, prefix):
    """Historical group Top3 rates as of previous DATE (never same-day)."""
    counts=(df.groupby([key,'date'],observed=True,dropna=False)['top3']
            .agg(['sum','count']).reset_index().sort_values([key,'date']))
    by=counts.groupby(key,sort=False,dropna=False)
    counts['prior_n']=by['count'].cumsum()-counts['count']
    counts['prior_w']=by['sum'].cumsum()-counts['sum']
    counts[prefix+'_n']=counts['prior_n'].astype('float32')
    counts[prefix+'_rate']=((counts['prior_w']+1)/(counts['prior_n']+3)).astype('float32')
    return df.merge(counts[[key,'date',prefix+'_n',prefix+'_rate']],on=[key,'date'],how='left',sort=False)

def build_features(raw):
    needed=['RaceDate','Course','RaceTime','HorseName','FPos','Ran','Type','Going','Yards','Class','WeightLBS','OR','Jockey','Trainer','Age']
    missing=[c for c in needed if c not in raw.columns]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    x=pd.DataFrame(index=raw.index)
    x['date']=pd.to_datetime(raw['RaceDate'],format='%d-%b-%y',errors='coerce')
    x['course']=clean_text(raw.Course)
    x['time']=raw.RaceTime.astype('string').fillna('')
    x['horse']=clean_text(raw.HorseName).str.replace(r'\s*\((ire|gb|fr|usa|ger|aus|nz)\)$','',regex=True)
    x['jockey']=clean_text(raw.Jockey)
    x['trainer']=clean_text(raw.Trainer)
    x['going']=norm_going(raw.Going)
    x['type']=clean_text(raw.Type)
    for src,dest in [('FPos','finish'),('Ran','field'),('Yards','yards'),('Class','class_num'),('WeightLBS','weight'),('OR','orating'),('Age','age')]:
        x[dest]=pd.to_numeric(raw[src],errors='coerce').astype('float32')
    x=x[x.date.notna() & x.horse.ne('') & x.course.ne('')].copy()
    x['race_id']=x['date'].dt.strftime('%Y%m%d')+'|'+x['course']+'|'+x['time']
    x['top3']=(x.finish.between(1,3)).astype('int8')
    x['valid_finish']=x.finish.between(1,x.field)
    x['finish_pct']=((x.field-x.finish)/(x.field-1)).where(x.valid_finish & (x.field>1))
    x=x.sort_values(['date','race_id','horse'],kind='stable').reset_index(drop=True)
    # A race's result can only be used after that race date. For horse-specific
    # history, group by horse/date to avoid same-day leakage too.
    daily=(x.groupby(['horse','date'],as_index=False,sort=True)
        .agg(finish=('finish','first'),finish_pct=('finish_pct','first'),top3=('top3','first'),
             class_num=('class_num','first'),weight=('weight','first'),orating=('orating','first'),
             field=('field','first'),yards=('yards','first')))
    daily=daily.sort_values(['horse','date'])
    g=daily.groupby('horse',sort=False)
    daily['prior_runs']=g.cumcount().astype('float32')
    for c in ['finish','finish_pct','class_num','weight','orating','field','yards']:
        daily['last_'+c]=g[c].shift(1).astype('float32')
    for n in (3,5,10):
        for c in ('top3','finish_pct'):
            shifted=g[c].shift(1)
            daily[f'last{n}_{c}']=shifted.groupby(daily.horse,sort=False).rolling(n,min_periods=1).mean().reset_index(level=0,drop=True).astype('float32')
    x=x.merge(daily.drop(columns=['finish','finish_pct','top3','class_num','weight','orating','field','yards']),on=['horse','date'],how='left',sort=False)
    x['class_change']=x['class_num']-x['last_class_num'] # positive = drop (3 -> 4)
    x['weight_change']=x['weight']-x['last_weight']
    x['or_change']=x['orating']-x['last_orating']
    x['distance_change']=abs(x['yards']-x['last_yards'])
    # History within the same course/going/type: only earlier calendar dates.
    for col,prefix in [('horse','horse'),('jockey','jockey'),('trainer','trainer')]:
        x=historical_rate(x,col,prefix)
    for c,prefix in [('course','course'),('going','going'),('type','type')]:
        key=prefix+'_horse_key'
        x[key]=x.horse+'|'+x[c]
        x=historical_rate(x,key,prefix)
        x.drop(columns=key,inplace=True)
    # Race-relative opposition strength uses ONLY pre-race estimates.
    x['opposition_form']=x.groupby('race_id')['last5_top3'].transform('mean').astype('float32')
    x['form_vs_field']=x['last5_top3']-x['opposition_form']
    x['or_vs_field']=x['orating']-x.groupby('race_id')['orating'].transform('mean')
    x['class_num']=x['class_num'].astype('float32')
    x['is_flat']=x['type'].isin(['f','a']).astype('float32')
    x['is_hurdle']=x['type'].eq('h').astype('float32')
    x['is_chase']=x['type'].eq('c').astype('float32')
    return x

FEATURES=['field','yards','class_num','weight','orating','age','prior_runs','last_finish',
'last_finish_pct','last_class_num','last_weight','last_orating','last_field','last_yards',
'last3_top3','last5_top3','last10_top3','last3_finish_pct','last5_finish_pct','last10_finish_pct',
'class_change','weight_change','or_change','distance_change',
'horse_n','horse_rate','jockey_n','jockey_rate','trainer_n','trainer_rate',
'course_n','course_rate','going_n','going_rate','type_n','type_rate',
'opposition_form','form_vs_field','or_vs_field','is_flat','is_hurdle','is_chase']

def run(csv,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(csv,low_memory=False)
    x=build_features(raw)
    train=x[(x.date<CUT)&x.valid_finish].copy()
    test=x[(x.date>=CUT)&(x.date<END)].copy()
    # Drop exact duplicate runner records, if any, before fitting/auditing.
    key=['date','course','time','horse']
    train=train.drop_duplicates(key)
    test=test.drop_duplicates(key)
    print(f'Training rows: {len(train):,}; test rows: {len(test):,}; test days: {test.date.nunique()}')
    model=HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=19,learning_rate=.055,l2_regularization=10,
                                          min_samples_leaf=80,random_state=42)
    model.fit(train[FEATURES].astype('float32'),train.top3)
    joblib.dump({'model':model,'features':FEATURES,'cutoff':'2026-08-31','target':'top3'},out/'v5_top3_model.joblib')
    test['pred_top3']=model.predict_proba(test[FEATURES].astype('float32'))[:,1]
    # One selection per race, to ensure Top 3 daily picks can form a standard treble.
    # NOTE: 'Ran' is the historical final field size; live deployment must use
    # declared runners from the pre-race card and separately handle non-runners.
    eligible=test[test.field.ge(6)].copy()  # pre-selection rule: minimum 6 runners; never filter on finishing position
    eligible=eligible.sort_values(['date','race_id','pred_top3','horse'],ascending=[True,True,False,True])
    eligible=eligible.drop_duplicates(['date','race_id'],keep='first')
    eligible=eligible.sort_values(['date','pred_top3','race_id'],ascending=[True,False,True])
    picks=eligible.groupby('date',sort=False).head(10).copy()
    picks['rank']=picks.groupby('date').cumcount()+1
    picks['placed']=picks.finish.between(1,3)
    if (picks.groupby('date').size() < 10).any():
        print('WARNING: Some test days have fewer than 10 eligible races.')
    picks['treble_leg']=picks['rank']<=3
    picks['result']=np.where(picks.finish.isna(),'UNKNOWN',np.where(picks.placed,'PLACED','LOST'))
    picks['place_odds']=np.nan  # Betfair prices need independent matching
    picks['race_date']=picks.date.dt.strftime('%Y-%m-%d')
    picks[['race_date','rank','course','time','horse','pred_top3','finish','field','class_num','going','placed','result','treble_leg','place_odds']].to_csv(out/'v5_daily_selections.csv',index=False)
    summary=(picks.groupby('race_date',sort=True).agg(selections=('horse','size'),placed=('placed','sum'),
                known=('finish','count'),treble_legs=('treble_leg','sum')).reset_index())
    treble=(picks[picks.treble_leg].groupby('race_date').agg(legs_placed=('placed','sum'),legs_known=('finish','count')).reset_index())
    summary=summary.merge(treble,on='race_date',how='left')
    summary['score']=summary.placed.astype(str)+'/'+summary.selections.astype(str)
    summary['treble_result']=np.select([(summary.legs_placed==3)&(summary.legs_known==3),
      (summary.legs_known-summary.legs_placed>0)],['WON','LOST'],default='PENDING')
    summary['treble_decimal_odds']=np.nan
    summary['treble_return']=np.nan
    summary.to_csv(out/'v5_historical_audit.csv',index=False)
    if test.valid_finish.any():
        z=test[test.valid_finish]
        print('September Brier score (all runners):',round(brier_score_loss(z.top3,z.pred_top3),5))
        print('September ROC AUC:',round(roc_auc_score(z.top3,z.pred_top3),5))
    print(summary[['race_date','score','treble_result']].to_string(index=False))
    print('Saved model and CSV audit in',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--out',default='v5_results')
    a=p.parse_args();run(a.csv,a.out)
