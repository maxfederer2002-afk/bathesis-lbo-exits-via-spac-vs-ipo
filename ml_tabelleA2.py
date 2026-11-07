"""
Tabelle A2 - Explorative Machine-Learning-Analyse (PyCaret)
Bachelorarbeit "SPAC vs. IPO als Exit-Kanal nach Leveraged Buyouts", Max Federer

Eigenstaendiges Skript: liest die Daten direkt aus der Arbeitsmappe,
gibt Tabelle A2 im Fenster aus und speichert die Panels als CSV sowie
ergebnisse_tabelle5.txt im Ordner dieses Skripts.

Zielgroessen: dLev = Lev(t+3) - Lev(t-1) und SGrowth (t-1 bis t+3)
Praediktoren: erklaerende Variablen der Gleichung (I)
Verfahren:    70/30-Aufteilung, zehnfache Kreuzvalidierung, Zufallsstartwert 0,
              Permutationswichtigkeit auf der Holdout-Stichprobe

Aufruf im Ordner dieses Skripts:
    python ml_tabelleA2.py
Voraussetzung: Python 3.11 mit den Paketen aus requirements.txt.
Die Arbeitsmappe muss im selben Ordner wie dieses Skript liegen.
Pakete:  Python 3.11, pycaret 3.3.2, scikit-learn 1.4.2, pandas, numpy, openpyxl
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
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pycaret.regression import (setup, compare_models, pull,
                                predict_model, get_config)
from sklearn.inspection import permutation_importance
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

SEED = 0
ZEILEN = []

def zeige(text=''):
    print(text)
    ZEILEN.append(str(text))

raw = lade_panel(XLSX)
raw['SPAC'] = (raw['kanal'] == 'De-SPAC').astype(int)
raw['Tech'] = (raw['ind'] == 'Technology').astype(int)
raw['Indu'] = (raw['ind'] == 'Industrials').astype(int)
raw['Heal'] = (raw['ind'] == 'Health Care').astype(int)
for v in ['Lev','LnAT','LnSale']:
    raw[v+'(t-1)'] = pd.to_numeric(raw[v+'_pre'], errors='coerce')
raw['Boom'] = pd.to_numeric(raw['Boom'], errors='coerce')
raw['dLev'] = (pd.to_numeric(raw['Lev_post'], errors='coerce')
               - pd.to_numeric(raw['Lev_pre'], errors='coerce'))
# SGrowth ist in der Auswertungsdatei bereits als kumuliertes Umsatzwachstum
# t-1 bis t+3 abgelegt (Spalte SGrowth_post), analog zu Tabelle 4 Spalte (3).
raw['SGrowth'] = pd.to_numeric(raw['SGrowth_post'], errors='coerce')

# Prädiktorsatz entspricht den erklaerenden Variablen der Gleichung (I):
# Exitkanal, Lev(t-1), Groesse, Boom und Branchencluster.
# EBITDAM(t-1) und piROA(t-1) sind entfernt, da sie in Gleichung (I)
# fuer dLev und SGrowth nicht enthalten sind.
FEATURES = ['SPAC','Lev(t-1)','LnAT(t-1)','LnSale(t-1)','Boom','Tech','Indu','Heal']


def lauf(target, beschreibung, panel_vgl, panel_imp, datei_vgl, datei_imp):
    """Ein kompletter PyCaret-Lauf fuer eine Zielgroesse.
    Prädiktorsatz ist fuer beide Zielgroessen identisch (FEATURES)."""
    data = raw[FEATURES + [target]].dropna().reset_index(drop=True)

    zeige('Zielgroesse: ' + beschreibung)
    zeige('Verfahren: PyCaret, zehnfache Kreuzvalidierung, Zufallsstartwert 0')
    zeige()
    zeige('Analysestichprobe: %d Unternehmen, davon %d De-SPAC'
          % (len(data), int(data['SPAC'].sum())))
    zeige()

    setup(data=data, target=target, session_id=SEED, fold=10,
          train_size=0.7, verbose=False, html=False)

    best = compare_models(include=['rf','lr','gbr','dt','dummy'], sort='MAE',
                          n_select=1, verbose=False)
    vgl = pull()

    zeige('PANEL %s - Modellvergleich' % panel_vgl)
    zeige(vgl[['Model','MAE','MSE','RMSE','R2','TT (Sec)']].to_string(index=False))
    zeige()
    zeige('Bestes Modell nach MAE: ' + type(best).__name__)
    zeige()

    # Holdout-Pruefung mit dem auf 70 Prozent trainierten Modell (best),
    # nicht mit dem nachtrainierten Modell, sonst waere der Holdout in-sample.
    _ = predict_model(best, verbose=False)
    zeige('Pruefung auf der Holdout-Stichprobe (30 Prozent)')
    zeige(pull()[['Model','MAE','MSE','RMSE','R2']].to_string(index=False))
    zeige()

    # Permutationswichtigkeit auf der Holdout-Stichprobe (30 Prozent),
    # damit sie den Beitrag zur Prognose an nicht verwendeten Unternehmen misst.
    X = get_config('X_test_transformed')
    y = get_config('y_test_transformed')
    imp = permutation_importance(best, X, y, n_repeats=50,
                                 random_state=SEED, scoring='r2')
    wicht = (pd.DataFrame({'Variable': X.columns, 'Wichtigkeit': imp.importances_mean})
             .sort_values('Wichtigkeit', ascending=False))
    zeige('PANEL %s - Variablenwichtigkeit (Permutationswichtigkeit)' % panel_imp)
    zeige(wicht.to_string(index=False))
    zeige()

    vgl[['Model','MAE','MSE','RMSE','R2','TT (Sec)']].to_csv(datei_vgl, index=False)
    wicht.to_csv(datei_imp, index=False)
    return datei_vgl, datei_imp


zeige('TABELLE A2 - Explorative Machine-Learning-Analyse')
zeige()

# Lauf 1: Verschuldungsgrad (Panels A und B)
a, b = lauf('dLev', 'dLev = Lev(t+3) - Lev(t-1)',
            'A', 'B', os.path.join(ORDNER, 'tabelle5_panelA.csv'), os.path.join(ORDNER, 'tabelle5_panelB.csv'))
zeige('=' * 70)
zeige()

# Lauf 2: kumuliertes Umsatzwachstum (Panel C; Panel D wird nicht berichtet)
c, d = lauf('SGrowth', 'SGrowth = kumuliertes Umsatzwachstum t-1 bis t+3',
            'C', 'D', os.path.join(ORDNER, 'tabelle5_panelC.csv'), os.path.join(ORDNER, 'tabelle5_panelD.csv'))

with open(os.path.join(ORDNER, 'ergebnisse_tabelle5.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(ZEILEN))

print('gespeichert: %s, %s, %s, %s, ergebnisse_tabelle5.txt' % (a, b, c, d))
