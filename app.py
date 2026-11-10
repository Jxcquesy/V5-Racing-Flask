"""V5 Flask replacement. Keeps the supplied frozen model and feature pipeline."""
import io, os, secrets, json, math, re, time
import joblib
from datetime import date, timedelta
from functools import wraps
from pathlib import Path
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, abort
from werkzeug.utils import secure_filename
from v5_engine import select_daily
from v5_fast_engine import predict_v5_fast
from pdf_parser import extract

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'v5_top3_model.joblib'
KEY=['RaceDate','Course','RaceTime','HorseName']
app=Flask(__name__)
app.secret_key=os.getenv('SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH']=49*1024*1024
# The session cookie holds only small state; master CSV is kept in a local instance file.
# On Render free hosting this file is EPHEMERAL and may disappear on redeploy/restart.
INSTANCE=Path(app.instance_path); INSTANCE.mkdir(parents=True,exist_ok=True)

def protected(fn):
    @wraps(fn)
    def inner(*args,**kwargs):
        if not os.getenv('APP_PASSWORD'):
            return render_template('setup.html',message='Set APP_PASSWORD in Render environment variables before using this app.'),503
        if not session.get('authenticated'):return redirect(url_for('login',next=request.path))
        return fn(*args,**kwargs)
    return inner

@app.get('/health')
def health():return {'status':'ok'}

@app.route('/login',methods=['GET','POST'])
def login():
    if not os.getenv('APP_PASSWORD'):return render_template('setup.html',message='APP_PASSWORD must be configured on Render.'),503
    if request.method=='POST':
        if secrets.compare_digest(request.form.get('password',''),os.getenv('APP_PASSWORD','')):
            session.clear();session['authenticated']=True
            return redirect(url_for('data'))
        flash('Incorrect password.','error')
    return render_template('login.html')

@app.post('/logout')
def logout():session.clear();return redirect(url_for('login'))

# Yearly CSVs live in private Supabase Storage; local files are disposable caches.
BUCKET=os.getenv('MASTER_BUCKET','v5-master')
YEAR_PATTERN=re.compile(r'^V5_Master_(20\d{2})\.csv$')
CACHE_SECONDS=600

def storage_config():
    url=os.getenv('SUPABASE_URL','').strip().rstrip('/')
    key=os.getenv('SUPABASE_SERVICE_ROLE_KEY','').strip()
    if not url.startswith('https://') or not key:
        raise ValueError('Configure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Render.')
    return url,key

def storage_headers(content_type=None):
    _,key=storage_config()
    h={'apikey':key,'Authorization':'Bearer '+key}
    if content_type:h['Content-Type']=content_type
    return h

def storage_years():
    url,_=storage_config()
    r=requests.post(f'{url}/storage/v1/object/list/{BUCKET}',headers=storage_headers('application/json'),json={'prefix':'','limit':1000,'offset':0},timeout=30)
    if not r.ok:raise RuntimeError(f'Cannot list private Storage bucket {BUCKET}: HTTP {r.status_code}. Create the bucket and upload yearly CSVs in Supabase.')
    return sorted(int(m.group(1)) for obj in r.json() if (m:=YEAR_PATTERN.fullmatch(obj.get('name',''))))

def yearly_path(year):return INSTANCE/f'V5_Master_{int(year)}.csv'

def download_year(year,force=False):
    p=yearly_path(year)
    if p.exists() and not force and time.time()-p.stat().st_mtime<CACHE_SECONDS:return p
    url,_=storage_config()
    r=requests.get(f'{url}/storage/v1/object/{BUCKET}/{p.name}',headers=storage_headers(),timeout=120,stream=True)
    if not r.ok:raise RuntimeError(f'Cannot download {p.name}: HTTP {r.status_code}')
    tmp=p.with_suffix('.tmp')
    try:
        with tmp.open('wb') as out:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:out.write(chunk)
        tmp.replace(p)
    finally:
        tmp.unlink(missing_ok=True)
    return p

def active_model_path():
    """Fetch the latest validated model from private Storage; never block fallback."""
    cached=INSTANCE/'v5_top3_model_active.joblib'
    if cached.exists() and time.time()-cached.stat().st_mtime<300:
        return cached
    tmp=cached.with_suffix('.tmp')
    try:
        url,_=storage_config()
        r=requests.get(f'{url}/storage/v1/object/{BUCKET}/models/v5_top3_model.joblib',
                       headers=storage_headers(),timeout=90,stream=True)
        if r.status_code==404:
            return cached if cached.exists() else MODEL
        r.raise_for_status()
        with tmp.open('wb') as out:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:out.write(chunk)
        candidate=joblib.load(tmp)
        baseline=joblib.load(MODEL)
        if candidate.get('features')!=baseline.get('features'):
            raise ValueError('Published model feature list does not match this V5 website.')
        if not hasattr(candidate.get('model'),'predict_proba'):
            raise ValueError('Published model is missing predict_proba.')
        tmp.replace(cached)
        return cached
    except Exception:
        return cached if cached.exists() else MODEL
    finally:
        tmp.unlink(missing_ok=True)

def training_status():
    try:
        url,_=storage_config()
        r=requests.get(f'{url}/storage/v1/object/{BUCKET}/models/training_status.json',
                       headers=storage_headers(),timeout=15)
        if not r.ok:return None
        return r.json()
    except Exception:
        return None

def upload_year(year,df):
    url,_=storage_config();p=yearly_path(year)
    tmp=p.with_suffix('.upload')
    df.to_csv(tmp,index=False)
    if tmp.stat().st_size>50*1024*1024:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'{year} exceeds 50 MB; this year must be subdivided before uploading.')
    headers=storage_headers('text/csv');headers['x-upsert']='true'
    try:
        with tmp.open('rb') as fh:
            r=requests.post(f'{url}/storage/v1/object/{BUCKET}/{p.name}',headers=headers,data=fh,timeout=180)
        if not r.ok:raise RuntimeError(f'Supabase Storage rejected {year} update: HTTP {r.status_code} {r.text[:250]}')
        tmp.replace(p)
    finally:tmp.unlink(missing_ok=True)

def norm(df):
    missing=[c for c in KEY if c not in df]
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    x=df.copy();dt=pd.to_datetime(x.RaceDate,errors='coerce',format='%d-%b-%y')
    if dt.isna().any():dt=pd.to_datetime(x.RaceDate,errors='coerce',dayfirst=True)
    if dt.isna().any():raise ValueError('Invalid RaceDate values')
    x['RaceDate']=dt.dt.strftime('%d-%b-%y');return x

def master():
    years=storage_years()
    if not years:return None
    return pd.concat([norm(pd.read_csv(download_year(y),low_memory=False)) for y in years],ignore_index=True)

def master_stats(years, missing_start=None, missing_end=None):
    """Count records and dates without materialising the entire Master."""
    total=0; recorded=set()
    for year in years:
        for part in pd.read_csv(download_year(year),usecols=['RaceDate'],chunksize=30000):
            total+=len(part)
            d=pd.to_datetime(part['RaceDate'],format='%d-%b-%y',errors='coerce')
            if d.isna().any():raise ValueError(f'Invalid RaceDate in {year} Master')
            recorded.update(d.dt.date.unique())
    if not recorded:return None,[]
    stats={'runners':total,'dates':len(recorded),'first':min(recorded).strftime('%d %b %Y'),
           'last':max(recorded).strftime('%d %b %Y')}
    missing=[]
    if missing_start is not None and missing_end is not None:
        start=pd.to_datetime(missing_start,errors='coerce')
        end=pd.to_datetime(missing_end,errors='coerce')
        if pd.notna(start) and pd.notna(end) and start<=end and (end-start).days<=1500:
            missing=[d.date().isoformat() for d in pd.date_range(start,end) if d.date() not in recorded]
    return stats,missing

def keys(df):
    return pd.MultiIndex.from_arrays([pd.to_datetime(df.RaceDate,format='%d-%b-%y').dt.strftime('%Y-%m-%d'),df.Course.astype(str).str.strip().str.casefold(),df.RaceTime.astype(str).str.strip(),df.HorseName.astype(str).str.strip().str.casefold()])

def supabase(method,payload=None,params=None):
    url=os.getenv('SUPABASE_URL','').strip().rstrip('/'); key=os.getenv('SUPABASE_SERVICE_ROLE_KEY','').strip()
    if not url.startswith('https://') or not key:raise ValueError('Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in Render environment variables.')
    if method not in ('GET','POST','PATCH'):raise ValueError('Unsupported method')
    headers={'apikey':key,'Authorization':'Bearer '+key,'Content-Type':'application/json','Prefer':'return=representation'}
    r=requests.request(method,url+'/rest/v1/v5_bets',headers=headers,json=payload,params=params,timeout=25)
    if not r.ok:raise RuntimeError(f'Supabase returned HTTP {r.status_code}')
    return r.json() if r.content else []

@app.route('/',methods=['GET','POST'])
@protected
def data():
    if request.method=='POST':
        try:
            action=request.form.get('action')
            if action=='merge':
                if request.form.get('verified')!='yes':raise ValueError('Confirm that you have verified these results.')
                f=request.files.get('results')
                if not f or not f.filename.lower().endswith('.csv'):raise ValueError('Choose a results CSV.')
                incoming=norm(pd.read_csv(f,low_memory=False))
                year_groups=incoming.groupby(pd.to_datetime(incoming.RaceDate,format='%d-%b-%y').dt.year)
                existing_years=set(storage_years())
                count=0
                for year,part in year_groups:
                    if year not in existing_years:raise ValueError(f'No existing {year} Master in Storage')
                    existing=norm(pd.read_csv(download_year(year),low_memory=False))
                    missing=[c for c in existing.columns if c not in part]
                    # Missing fields may be left empty; required identity fields were checked by norm().
                    part=part.reindex(columns=existing.columns)
                    k=keys(part)
                    new=part.loc[~k.isin(keys(existing)) & ~k.duplicated()].copy()
                    if len(new):
                        combined=pd.concat([existing,new],ignore_index=True)
                        upload_year(year,combined)
                        count+=len(new)
                    del existing,part,new
                flash(f'Merged {count:,} new runner records into permanent yearly Supabase Storage.','success')
            else:raise ValueError('Unknown action')
        except Exception as e:flash(str(e),'error')
        return redirect(url_for('data'))
    stats=None;missing=[];storage_error=None;years=[]
    try:
        years=storage_years()
        if years:
            stats,missing=master_stats(years,
                request.args.get('start') if request.args.get('missing')=='1' else None,
                request.args.get('end') if request.args.get('missing')=='1' else None)
    except Exception as e:storage_error=str(e)
    return render_template('data.html',stats=stats,missing=missing,today=date.today().isoformat(),years=years,storage_error=storage_error,bucket=BUCKET,training_status=training_status())

@app.get('/master/download')
@protected
def download_master():
    # Stream existing year files without a second full in-memory DataFrame.
    from flask import Response, stream_with_context
    import csv
    years=storage_years()
    if not years:raise ValueError('No Master CSVs found')
    def generate():
        first=True
        for year in years:
            with open(download_year(year),'r',encoding='utf-8-sig',newline='') as fh:
                if first:
                    yield fh.readline();first=False
                else:fh.readline()
                for line in fh:yield line
    return Response(stream_with_context(generate()),mimetype='text/csv',headers={
        'Content-Disposition':'attachment; filename="V5_Master_UPDATED.csv"'})

class PDFUpload:
    def __init__(self,f):self.name=secure_filename(f.filename);self._bytes=f.read()
    def getvalue(self):return self._bytes

@app.route('/selections',methods=['GET','POST'])
@protected
def selections():
    picks=session.get('picks',[])
    if request.method=='POST':
        try:
            day=date.fromisoformat(request.form.get('race_date',''))
            csv=request.files.get('runners');pdfs=[f for f in request.files.getlist('pdfs') if f and f.filename]
            if csv and csv.filename:
                if not csv.filename.lower().endswith('.csv'):raise ValueError('Racecard file must be CSV.')
                runners=pd.read_csv(io.BytesIO(csv.read()))
            elif pdfs:
                if any(not f.filename.lower().endswith('.pdf') for f in pdfs):raise ValueError('Racecards must be PDF files.')
                runners,issues=extract([PDFUpload(f) for f in pdfs],day)
                for issue in issues:flash(issue,'error')
            else:raise ValueError('Upload a corrected racecard CSV or PDF racecards.')
            if runners.empty:raise ValueError('No racecard runners found.')
            if request.form.get('verified')!='yes':raise ValueError('Please confirm you have checked the racecards.')
            for col in ('Jockey','Trainer','Distance'):
                if col not in runners:runners[col]=''
            year_files=[download_year(y) for y in storage_years()]
            if not year_files:raise ValueError("Upload yearly Master CSVs first")
            predictions=predict_v5_fast(year_files,runners,active_model_path(),day)
            selected=select_daily(predictions).copy()
            selected['_sort']=pd.to_datetime(selected.Time.astype(str),format='%H:%M',errors='coerce')
            selected=selected.sort_values(['_sort','Course','Horse'],kind='stable').drop(columns='_sort')
            picks=[]
            for p in selected.to_dict('records'):
                prob=float(p['Top 3 Probability'])
                picks.append({'horse':str(p['Horse']),'course':str(p['Course']),'time':str(p['Time']),'probability':prob,'min_odds':1.05/prob if prob>0 else None,'race_date':day.isoformat()})
            session['picks']=picks;session['pick_index']=0
            flash(f'{len(picks)} V5 selections generated.','success')
            return redirect(url_for('selections'))
        except Exception as e:flash('Prediction failed: '+str(e),'error')
    picks=session.get('picks',[]);i=max(0,min(int(request.args.get('i',session.get('pick_index',0))),max(0,len(picks)-1)))
    return render_template('selections.html',picks=picks,index=i,today=date.today().isoformat(),master_ready=bool(storage_years()),slip_count=len(session.get('slip',[])))

@app.post('/slip/add')
@protected
def add_leg():
    picks=session.get('picks',[])
    try:
        i=int(request.form.get('index','-1'));p=picks[i]
        if i<0:raise ValueError('Invalid selection')
        odds=float(request.form.get('odds',''))
        if not 1.01<=odds<=1001:raise ValueError('Odds must be between 1.01 and 1001')
        slip=session.get('slip',[]);key=(p['race_date'],p['course'],p['time'],p['horse'])
        slip=[x for x in slip if (x['race_date'],x['course'],x['time'],x['horse'])!=key]
        slip.append({**p,'odds':odds});session['slip']=slip
        session['pick_index']=min(i+1,len(picks)-1)
        flash('Horse added to betslip.','success')
    except Exception as e:flash(str(e),'error')
    return redirect(url_for('selections'))

@app.route('/betslip',methods=['GET','POST'])
@protected
def betslip():
    slip=session.get('slip',[])
    if request.method=='POST':
        try:
            action=request.form.get('action')
            if action=='remove':
                i=int(request.form['index']);slip.pop(i);session['slip']=slip;flash('Removed selection.','success')
            elif action=='save':
                if not slip:raise ValueError('Betslip is empty')
                days={x['race_date'] for x in slip}
                if len(days)!=1:raise ValueError('Save separate slips for different race dates.')
                units=float(request.form.get('units','1'));unit_value=float(request.form.get('unit_value','.4'))
                if not (0<units<=100000 and 0<unit_value<=100000):raise ValueError('Invalid stake')
                combined=math.prod(float(x['odds']) for x in slip)
                legs=[{k:v for k,v in leg.items() if k!='race_date'} for leg in slip]
                # Existing Supabase schema has no unit_value column. Keep compatible with existing history.
                supabase('POST',{'race_date':next(iter(days)),'legs':legs,'combined_odds':round(combined,6),'stake_units':units,'result':'Pending'})
                session['slip']=[];flash('Saved to your existing Supabase betting history.','success')
            else:raise ValueError('Unknown action')
        except Exception as e:flash(str(e),'error')
        return redirect(url_for('betslip'))
    combined=math.prod(float(x['odds']) for x in slip)
    return render_template('betslip.html',slip=slip,combined=combined)

@app.route('/history',methods=['GET','POST'])
@protected
def history():
    if request.method=='POST':
        try:
            id_=request.form.get('id','');status=request.form.get('result','')
            if not id_ or status not in ('Pending','Won','Lost','Void'):raise ValueError('Invalid result')
            supabase('PATCH',{'result':status},{'id':'eq.'+id_});flash('Result updated.','success')
        except Exception as e:flash(str(e),'error')
        return redirect(url_for('history'))
    rows=[];error=None;pnl=0.0
    try:
        rows=supabase('GET',params={'select':'*','order':'created_at.desc','limit':'1000'})
        for r in rows:
            stake=float(r.get('stake_units') or 0);status=r.get('result')
            if status=='Won':pnl+=(float(r.get('combined_odds') or 1)-1)*stake
            elif status=='Lost':pnl-=stake
    except Exception as e:error=str(e)
    return render_template('history.html',rows=rows,pnl=pnl,error=error)

if __name__=='__main__':app.run(debug=False)