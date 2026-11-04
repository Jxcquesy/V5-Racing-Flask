"""Best-effort At The Races PDF extraction; always review before predicting."""
import re

import fitz
import pandas as pd


PEDIGREE = re.compile(r'^\s*(?:b|br|ch|gr|g|dk b|ro|bl)\s+[cfgm]\s+.+\s+-\s+.+', re.I)
WEIGHT = re.compile(r'^\s*(\d{1,2})\s+(\d{1,2})\s*-\s*(\d{1,2})')
RATING = re.compile(r'^(?:\d{2,3}|-)$')
HEADERS = re.compile(r'\(R(\d+)\)\s*(\d{1,2}:\d{2})\s+([A-Z][A-Z .()\-]+?),\s*([^\n]+)')


def _clean_horse(name):
    name = name.strip()
    name = re.sub(r'^\s*[0-9PFRU/\-]{2,12}\s+(?=[A-Z])', '', name)
    # PDF racecards sometimes append days since last run and C/D/BF indicators.
    name = re.sub(r'\s+\d{1,4}(?:\s+\(\d+F\))?(?:\s+(?:CD|C|D|BF)){0,4}\s*$', '', name, flags=re.I)
    name = re.sub(r'(?:\s+(?:CD|C|D|BF))+\s*$', '', name, flags=re.I)
    return name.strip()


def _connections(lines, weight_index):
    """Read lines between the weight and the OR; last line is the trainer.

    A jockey name and their claim can wrap onto multiple lines. Returns
    blanks when the PDF does not contain an unambiguous pair.
    """
    parts = []
    rating = ''
    found_rating = False
    for raw in lines[weight_index + 1:weight_index + 13]:
        item = raw.strip()
        if item.startswith('Jockey Colours:'):
            break
        if not item:
            continue
        if RATING.fullmatch(item):
            rating = '' if item == '-' else item
            found_rating = True
            break
        parts.append(item)
    if not found_rating or len(parts) < 2:
        return '', '', rating
    # A trainer's name can wrap over two lines, e.g.:
    # Jack Callan / Dr R Newland & J / Insole / 75
    # Only merge when the second line clearly begins a trainer name.
    trainer_start = re.compile(
        r'^(?:Dr\b|Sir\b|Mrs\b|Miss\b|Ms\b|Mr\b|Prof\b)', re.I
    )
    if len(parts) >= 3 and trainer_start.match(parts[1]):
        return parts[0], ' '.join(parts[1:]), rating
    return ' '.join(parts[:-1]), parts[-1], rating


def extract(files, date):
    rows = []
    warnings = []
    for file in files:
        doc = fitz.open(stream=file.getvalue(), filetype='pdf')
        try:
            full = '\n'.join(page.get_text() for page in doc)
        finally:
            doc.close()
        headers = list(HEADERS.finditer(full))
        if not headers:
            warnings.append(f'{file.name}: no recognised race headings')
        for hi, m in enumerate(headers):
            course = re.sub(r'\s*\(A\.?W\.?\)\s*$', '', m.group(3), flags=re.I).strip().title()
            text = full[m.end():headers[hi + 1].start() if hi + 1 < len(headers) else len(full)]
            intro = text.split('No(Dr)', 1)[0][:700].replace('\n', ' ')
            cl = re.search(r'\(Class\s*(\d+)\)', intro)
            typ = (
                'NH Flat' if re.search(r'\b(?:National Hunt Flat|NH Flat|Bumper|Flat Race)\b', intro, re.I) else
                'Chase' if re.search(r'\bChase\b', intro, re.I) else
                'Hurdle' if re.search(r'\bHurdle\b', intro, re.I) else
                'Flat' if re.search(r'\b(?:Nursery|Handicap|Maiden|Novice|Stakes|Conditions|Claiming|Selling)\b', intro, re.I) or '(A.W.)' in m.group(3) else ''
            )
            seen = set()
            count = 0
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if not PEDIGREE.match(line.strip()) or i == 0 or i + 1 >= len(lines):
                    continue
                horse = _clean_horse(lines[i - 1])
                if horse in ('D', 'BF', 'C', 'CD') or not re.match(r'^[A-Z]', horse) or len(horse) > 65 or horse in seen:
                    continue
                # The weight is normally the next line; preserve original
                # look-ahead for PDF variations.
                weight_index = None
                wm = None
                for j in range(i + 1, min(i + 4, len(lines))):
                    wm = WEIGHT.match(lines[j])
                    if wm:
                        weight_index = j
                        break
                if weight_index is None:
                    continue
                age, stone, lbs = map(int, wm.groups())
                if not (2 <= age <= 20 and 9 <= stone <= 12 and 0 <= lbs <= 13):
                    continue
                jockey, trainer, orating = _connections(lines, weight_index)
                if not jockey or not trainer:
                    warnings.append(f'{course} {m.group(2)} {horse}: jockey/trainer needs review')
                seen.add(horse)
                count += 1
                rows.append(dict(
                    Horse=horse, Course=course, Time=m.group(2),
                    Final_Race_Type=typ, Final_Going='', RFB_Age=age,
                    RFB_WeightLBS=stone * 14 + lbs, RFB_OR=orating,
                    Final_Class=cl.group(1) if cl else '', RaceDate=str(date),
                    Distance=m.group(4).strip(), RaceNumber=m.group(1),
                    Jockey=jockey, Trainer=trainer,
                ))
            if count == 0:
                warnings.append(f'{course} {m.group(2)}: no runners extracted')
            if not typ:
                warnings.append(f'{course} {m.group(2)}: race type needs review')
    return pd.DataFrame(rows), warnings
