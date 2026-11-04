"""V5 race-specific going lookup using Racing Post's publicly returned page data.

Review matches before predicting. Website structure and access can change.
"""
import json
import re
from datetime import date as Date

import pandas as pd

URL = 'https://www.racingpost.com/non-runners/'


def course_key(value):
    s = str(value).lower().strip()
    s = re.sub(r'\s*\(a\.?w\.?\)', '', s)
    s = re.sub(r'\b(?:park|city)\b', '', s)
    return re.sub(r'[^a-z0-9]', '', s)


def time_key(value):
    s = str(value).strip()
    m = re.match(r'^(\d{1,2}):(\d{2})(?::\d{2})?$', s)
    if not m:
        return None
    h, minute = map(int, m.groups())
    if h > 23 or minute > 59:
        return None
    return f'{h:02d}:{minute:02d}'


def fetch_races(race_date, timeout=15):
    """Return (race dataframe, error). Fail closed if requested date absent."""
    import requests
    from bs4 import BeautifulSoup

    wanted = str(race_date)[:10]
    try:
        response = requests.get(URL, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; personal-racecard-review/1.0)'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        script = soup.find('script', id='__NEXT_DATA__')
        if script is None:
            return pd.DataFrame(), 'Racing Post structured data was not found.'
        state = json.loads(script.get_text())['props']['pageProps']['initialState']
        dates = state['meetings']['byDate']
        if wanted not in dates:
            return pd.DataFrame(), f'No report for {wanted}; available dates: {", ".join(dates.keys())}.'
        day = dates[wanted]
        races = day.get('races', {}).get('byRaceId', {})
        rows = []
        for rid, race in races.items():
            course = race.get('meetingName') or race.get('diffusionMeetingName') or ''
            start = race.get('startTime')
            going = race.get('going') or ''
            if not course or not time_key(start):
                continue
            rows.append({
                'RaceDate': wanted, 'Course': course, 'Time': time_key(start),
                'RaceType': race.get('raceType') or '',
                'Surface': race.get('surfaceType') or '',
                'Going': going, 'Abandoned': bool(race.get('isAbandoned')),
                'RaceID': str(rid),
            })
        if not rows:
            return pd.DataFrame(), 'No race records with course and time found.'
        return pd.DataFrame(rows), ''
    except Exception as exc:
        return pd.DataFrame(), f'Going lookup unavailable ({type(exc).__name__}: {exc}).'


def apply_going(runners, races, race_date):
    """Match by exact date, normalised course and scheduled time.

    Only fills empty Final_Going; ambiguous/abandoned/unmatched races are flagged.
    Returns (updated runners, audit dataframe).
    """
    df = runners.copy()
    if 'Final_Going' not in df.columns:
        df['Final_Going'] = ''
    wanted = str(race_date)[:10]
    lookup = {}
    for _, race in races.iterrows():
        if str(race.get('RaceDate', ''))[:10] != wanted:
            continue
        key = (course_key(race.get('Course', '')), time_key(race.get('Time', '')))
        if key[0] and key[1]:
            lookup.setdefault(key, []).append(race)
    audit = []
    for idx, runner in df.iterrows():
        key = (course_key(runner.get('Course', '')), time_key(runner.get('Time', '')))
        matched = lookup.get(key, [])
        old = runner.get('Final_Going', '')
        if pd.notna(old) and str(old).strip():
            status = 'Existing value kept'
        elif len(matched) != 1:
            status = 'Unmatched' if not matched else 'Ambiguous race match'
        elif bool(matched[0].get('Abandoned')):
            status = 'Abandoned - do not predict'
        elif not str(matched[0].get('Going', '')).strip():
            status = 'Missing going - review'
        else:
            df.at[idx, 'Final_Going'] = str(matched[0]['Going']).strip()
            status = 'Matched - review'
        audit.append({
            'Course': runner.get('Course', ''), 'Time': runner.get('Time', ''),
            'Horse': runner.get('Horse', ''), 'Going': df.at[idx, 'Final_Going'],
            'Status': status,
        })
    return df, pd.DataFrame(audit)
