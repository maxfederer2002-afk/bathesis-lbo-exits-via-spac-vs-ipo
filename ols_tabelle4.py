"""
Tabelle 4 - Multivariate Veraenderungsregressionen (OLS) und Medianregression
Bachelorarbeit "SPAC vs. IPO als Exit-Kanal nach Leveraged Buyouts", Max Federer

Eigenstaendiges Skript: liest die Daten direkt aus der Arbeitsmappe und
gibt Tabelle 4 im Fenster aus. Es werden keine Dateien geschrieben.

Spezifikation (Gleichung I):
    dY(i) = b0 + b1*SPAC + b2*Y(i,t-1) + b3*LnAT(t-1) + b4*LnSale(t-1)
            + b5*Lev(t-1) + b6*Boom + Branchencluster + e(i)
    mit dY = Y(t+3) - Y(t-1)
Panel B (Gleichung II) erweitert diese Spezifikation um Gross und SPAC x Gross.
Zusaetzlich: Medianregression (q = 0,5) als Ausreisserpruefung.

- Ausreisserbehandlung: keine (WINSOR = False)
- Standardfehler: heteroskedastizitaetsrobust nach White, HC3
- Referenzbranche: Uebrige Cluster (ohne Financials und Real Estate)

Aufruf im Ordner dieses Skripts:
    python ols_tabelle4.py
Voraussetzung: Python 3.11 mit den Paketen aus requirements.txt.
Die Arbeitsmappe muss im selben Ordner wie dieses Skript liegen.
Pakete:  pandas, numpy, statsmodels, openpyxl
"""
# ---------------------------------------------------------------------------
# Start ueber die virtuelle Umgebung: Liegt eine .venv in diesem Ordner oder im
# Ordner darueber, startet sich das Skript automatisch mit deren Python neu.
# So genuegt "python <skript>.py", ohne Pfade und ohne Aktivierung der .venv.
# ---------------------------------------------------------------------------
import os
import sys
import subprocess

_HIER = os.path.dirname(os.path.abspath(__file__))
if os.environ.get('VENV_NEUSTART') != '1':
    for _kandidat in ('.venv', os.path.join('..', '.venv')):
        for _exe in (os.path.join('Scripts', 'python.exe'), os.path.join('bin', 'python')):
            _py = os.path.normpath(os.path.join(_HIER, _kandidat, _exe))
            if os.path.isfile(_py) and os.path.normcase(_py) != os.path.normcase(sys.executable):
                os.environ['VENV_NEUSTART'] = '1'
                sys.exit(subprocess.call([_py, os.path.abspath(__file__)] + sys.argv[1:]))
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
# ---------------------------------------------------------------------------
# Datenquelle: Blatt "Panel" der Arbeitsmappe Statistische_Auswertungen_und_Tables.xlsx
# im selben Ordner wie dieses Skript. Ein Pfad als Argument ueberschreibt sie:
#     python <skript>.py pfad\zur\arbeitsmappe.xlsx
# Gelesen werden nur gespeicherte Werte (data_only=True), keine Formeln.
#
# Filter (identisch zu Tabelle 2 und Tabelle 3):
#   - Spalte B (Kanal) ist "IPO" oder "De-SPAC"
#   - Spalte D (ICB Industry) ist nicht "Financials" und nicht "Real Estate"
#   - Spalte AV (Flag: Daten t-1 bis t+3 vollstaendig) ist 1
# Die Variablenbloecke werden ueber Kopfzeile 1 und die Periodenoffsets in
# Zeile 2 gesucht.
# ---------------------------------------------------------------------------
import io
import os
import sys
import csv
import openpyxl
import pandas as pd

ORDNER = os.path.dirname(os.path.abspath(__file__))
# Nur ein Argument mit Endung .xlsx gilt als Pfad. So laeuft das Skript auch
# aus einer Python-Konsole, die eigene Eintraege in sys.argv setzt.
_ARG = [a for a in sys.argv[1:] if a.lower().endswith('.xlsx')]
XLSX = (os.path.abspath(_ARG[0]) if _ARG
        else os.path.join(ORDNER, 'Statistische_Auswertungen_und_Tables.xlsx'))

BLOECKE = ['EBITDAM', 'piROS', 'piROA', 'SGrowth', 'Lev', 'NDEBITDA', 'IntCov',
           'BVE', 'LnAT', 'LnSale']


def lade_panel(pfad):
    """Liest das Blatt Panel und gibt die Analysestichprobe als DataFrame zurueck."""
    if not os.path.isfile(pfad):
        raise SystemExit(f'Arbeitsmappe nicht gefunden: {pfad}')
    ws = openpyxl.load_workbook(pfad, data_only=True)['Panel']

    start, einzel = {}, {}
    for c in range(1, ws.max_column + 1):
        name = ws.cell(1, c).value
        if isinstance(name, str) and name.strip():
            name = name.strip()
            if name in BLOECKE and name not in start:
                start[name] = c
            einzel.setdefault(name, c)
    spalten = {}
    for k in BLOECKE:
        if k not in start:
            raise SystemExit(f'Spaltenblock "{k}" nicht gefunden')
        pre = post = None
        for c in range(start[k], start[k] + 6):
            if ws.cell(2, c).value == -1:
                pre = c
            elif ws.cell(2, c).value == 3:
                post = c
                break
        if pre is None or post is None:
            raise SystemExit(f'Perioden -1/+3 im Block "{k}" nicht gefunden')
        spalten[k] = (pre, post)
    for k in ('DealVal', 'Boom'):
        if k not in einzel:
            raise SystemExit(f'Spalte "{k}" nicht gefunden')

    felder = ['name', 'kanal', 'ind', 'jahr', 'DealVal', 'Boom']
    for k in spalten:
        felder += [k + '_pre', k + '_post']

    zeilen = []
    for r in range(3, ws.max_row + 1):
        kanal = ws[f'B{r}'].value
        if kanal not in ('IPO', 'De-SPAC'):
            continue
        if ws[f'D{r}'].value in ('Financials', 'Real Estate'):
            continue
        if ws[f'AV{r}'].value != 1:
            continue
        d = {'name': ws[f'A{r}'].value, 'kanal': kanal, 'ind': ws[f'D{r}'].value,
             'jahr': ws[f'C{r}'].value,
             'DealVal': ws.cell(r, einzel['DealVal']).value,
             'Boom': ws.cell(r, einzel['Boom']).value}
        for k, (a, b) in spalten.items():
            va, vb = ws.cell(r, a).value, ws.cell(r, b).value
            d[k + '_pre'] = va if isinstance(va, (int, float)) else ''
            d[k + '_post'] = vb if isinstance(vb, (int, float)) else ''
        zeilen.append(d)

    # Umweg ueber einen CSV-Puffer im Arbeitsspeicher, damit die Datentypen
    # exakt so eingelesen werden wie frueher aus regressionsdaten.csv.
    puffer = io.StringIO()
    w = csv.DictWriter(puffer, fieldnames=felder)
    w.writeheader()
    w.writerows(zeilen)
    puffer.seek(0)
    df = pd.read_csv(puffer)
    n_spac = int((df['kanal'] == 'De-SPAC').sum())
    print(f'Datenquelle: {pfad}')
    print(f'Analysestichprobe: {len(df)} Unternehmen ({n_spac} De-SPAC, {len(df) - n_spac} IPO)')
    print()
    return df


df = pd.DataFrame(lade_panel(XLSX).to_dict('records'))
df['SPAC'] = (df['kanal'] == 'De-SPAC').astype(float)
df['Tech'] = (df['ind'] == 'Technology').astype(float)
df['Indu'] = (df['ind'] == 'Industrials').astype(float)
df['Heal'] = (df['ind'] == 'Health Care').astype(float)


WINSOR = False   # Hauptspezifikation ohne Winsorisierung


def winsor(s, lo=0.01, hi=0.99):
    s = pd.to_numeric(s, errors='coerce')
    if not WINSOR or s.notna().sum() < 5:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


# Niveaus t-1 winsorisieren (Kontrollen)
for v in ['EBITDAM', 'piROA', 'Lev', 'NDEBITDA', 'IntCov', 'LnAT', 'LnSale']:
    df[v + '_pre_w'] = winsor(df[v + '_pre'])

# Abhaengige Variablen: Veraenderung t-1 bis t+3, danach winsorisiert
DEPS = [
    ('dEBITDAM', 'EBITDAM', 'EBITDAM_pre_w', 'neg'),
    ('dpiROA',   'piROA',   'piROA_pre_w',   'neg'),
    ('SGrowth',  None,      None,            'pos'),
    ('dLev',     'Lev',     'Lev_pre_w',     'pos'),
    ('dNDEBITDA', 'NDEBITDA', 'NDEBITDA_pre_w', 'pos'),
    ('dIntCov',  'IntCov',  'IntCov_pre_w',  'neg'),
]
for name, base, _, _ in DEPS:
    if base is None:
        df[name] = winsor(df['SGrowth_post'])
    else:
        d = pd.to_numeric(df[base + '_post'], errors='coerce') - \
            pd.to_numeric(df[base + '_pre'], errors='coerce')
        df[name] = winsor(d)

df['Gross'] = (df['LnSale_pre_w'] >= df['LnSale_pre_w'].median()).astype(float)

BASE = ['SPAC', 'LnAT_pre_w', 'LnSale_pre_w', 'Lev_pre_w', 'Boom', 'Tech', 'Indu', 'Heal']


def fit(dep, lagged, extra=None):
    regs = list(BASE)
    if lagged and lagged not in regs:
        regs.insert(1, lagged)
    if extra:
        regs += extra
    d = df[[dep] + regs].apply(pd.to_numeric, errors='coerce').dropna()
    X = sm.add_constant(d[regs], has_constant='add')
    return sm.OLS(d[dep], X).fit(cov_type='HC3'), d, regs


def one_sided(res, direction):
    b = res.params['SPAC']
    p2 = res.pvalues['SPAC']
    ok = (b < 0) if direction == 'neg' else (b > 0)
    return p2 / 2 if ok else 1 - p2 / 2


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(idx):
        val = (m - k) * pvals[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    return adj


results, praw = {}, []
for dep, base, lagged, direction in DEPS:
    res, d, regs = fit(dep, lagged)
    med = smf.quantreg(f"{dep} ~ " + " + ".join(regs), d).fit(q=0.5)
    results[dep] = {'res': res, 'regs': regs, 'lagged': lagged, 'n': int(res.nobs), 'med': med,
                    'p1': one_sided(res, direction), 'dir': direction,
                    'med_b': med.params['SPAC'], 'med_p': med.pvalues['SPAC']}
    praw.append(results[dep]['p1'])

for dep, adj in zip([d[0] for d in DEPS], holm(np.array(praw))):
    results[dep]['p_holm'] = adj

# Panel B: Moderation nach Unternehmensgroesse
mod = {}
for dep, base, lagged, direction in DEPS:
    df['SPACxGross'] = df['SPAC'] * df['Gross']
    res, d, regs = fit(dep, lagged, extra=['Gross', 'SPACxGross'])
    mod[dep] = res


def star(p):
    return '***' if p < .01 else '**' if p < .05 else '*' if p < .10 else ''


print('=' * 100)
print('PANEL A  -  Basismodelle (HC3), zweiseitige p-Werte')
print('=' * 100)
hdr = f"{'':<16}" + ''.join(f"{d[0]:>14}" for d in DEPS)
print(hdr)
rows = ['SPAC', 'LAG', 'LnAT_pre_w', 'LnSale_pre_w', 'Lev_pre_w', 'Boom', 'const']
label = {'LAG': 'Y(t-1)', 'LnAT_pre_w': 'LnAT(t-1)', 'LnSale_pre_w': 'LnSale(t-1)',
         'Lev_pre_w': 'Lev(t-1)', 'const': 'Konstante'}
for rn in rows:
    line = f"{label.get(rn, rn):<16}"
    line2 = f"{'':<16}"
    for dep, base, lagged, direction in DEPS:
        r = results[dep]['res']
        key = results[dep]['lagged'] if rn == 'LAG' else rn
        if rn == 'LAG' and (key is None or key == 'Lev_pre_w'):
            line += f"{'-':>14}"; line2 += f"{'':>14}"; continue
        if rn == 'Lev_pre_w' and results[dep]['lagged'] == 'Lev_pre_w':
            pass
        if key not in r.params:
            line += f"{'-':>14}"; line2 += f"{'':>14}"; continue
        p = r.pvalues[key]
        line += f"{r.params[key]:>11.3f}{star(p):<3}"
        line2 += f"{'(' + format(p, '.3f') + ')':>14}"
    print(line); print(line2)
print(f"{'Branchencluster':<16}" + ''.join(f"{'Ja':>14}" for _ in DEPS))
print(f"{'N':<16}" + ''.join(f"{results[d[0]]['n']:>14}" for d in DEPS))
print(f"{'Adj. R2':<16}" + ''.join(f"{results[d[0]]['res'].rsquared_adj:>14.3f}" for d in DEPS))
print(f"{'p Holm (SPAC)':<16}" + ''.join(f"{results[d[0]]['p_holm']:>14.3f}" for d in DEPS))
print(f"{'Median-Reg b1':<16}" + ''.join(f"{results[d[0]]['med_b']:>14.3f}" for d in DEPS))
print(f"{'Median-Reg p':<16}" + ''.join(f"{results[d[0]]['med_p']:>14.3f}" for d in DEPS))

print()
print('=' * 100)
print('PANEL B  -  Moderation nach Unternehmensgroesse (zweiseitige p-Werte)')
print('=' * 100)
print(hdr)
for rn in ['SPAC', 'Gross', 'SPACxGross']:
    line = f"{rn:<16}"; line2 = f"{'':<16}"
    for dep, *_ in DEPS:
        r = mod[dep]
        if rn in r.params:
            line += f"{r.params[rn]:>11.3f}{star(r.pvalues[rn]):<3}"
            line2 += f"{'(' + format(r.pvalues[rn], '.3f') + ')':>14}"
        else:
            line += f"{'-':>14}"; line2 += f"{'':>14}"
    print(line); print(line2)
print(f"{'N':<16}" + ''.join(f"{int(mod[d[0]].nobs):>14}" for d in DEPS))
print(f"{'Adj. R2':<16}" + ''.join(f"{mod[d[0]].rsquared_adj:>14.3f}" for d in DEPS))

print()
print('=' * 100)
print('MEDIANREGRESSION  -  Quantilsregression q = 0,5, alle Koeffizienten (zweiseitig)')
print('=' * 100)
print(hdr)
for rn in ['SPAC', 'LAG', 'LnAT_pre_w', 'LnSale_pre_w', 'Lev_pre_w', 'Boom',
           'Tech', 'Indu', 'Heal', 'Intercept']:
    line = f"{label.get(rn, 'Konstante' if rn == 'Intercept' else rn):<16}"
    line2 = f"{'':<16}"
    for dep, base, lagged, direction in DEPS:
        m = results[dep]['med']
        key = results[dep]['lagged'] if rn == 'LAG' else rn
        if (rn == 'LAG' and (key is None or key == 'Lev_pre_w')) or key not in m.params:
            line += f"{'-':>14}"; line2 += f"{'':>14}"; continue
        line += f"{m.params[key]:>11.3f}{star(m.pvalues[key]):<3}"
        line2 += f"{'(' + format(m.pvalues[key], '.3f') + ')':>14}"
    print(line); print(line2)
print(f"{'Pseudo R2':<16}" + ''.join(f"{results[d[0]]['med'].prsquared:>14.3f}" for d in DEPS))
print(f"{'N':<16}" + ''.join(f"{int(results[d[0]]['med'].nobs):>14}" for d in DEPS))
