# SPAC vs. IPO as an exit channel after leveraged buyouts

Replication code for the bachelor's thesis "SPAC vs. IPO als Exit-Kanal nach Leveraged Buyouts.
Eine quantitative Analyse von Performance und Kapitalstruktur PE-gesponserter US-Unternehmen"
by Max Federer, Frankfurt School of Finance & Management, 2026.

## Contents

| File | Purpose |
|---|---|
| `ols_tabelle4.py` | Table 4: OLS change regressions with HC3 standard errors and median regression |
| `ml_tabelleA2.py` | Table A2: exploratory machine learning analysis with PyCaret |
| `requirements.txt` | Package versions used |

Both scripts read the sheet "Panel" of the workbook `Statistische_Auswertungen_und_Tables.xlsx`,
which must be placed in the same folder as the scripts. They apply the sample filters of the thesis
(exit via IPO or De-SPAC, excluding Financials and Real Estate, complete data from t-1 to t+3).

## Requirements

Python 3.11 with the packages listed in `requirements.txt`. The analyses for the thesis were run
locally on Windows in PowerShell, using a virtual environment with these package versions.

## Usage (Windows PowerShell, in the folder of the scripts)

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe ols_tabelle4.py
    .\.venv\Scripts\python.exe ml_tabelleA2.py

If a `.venv` exists in the script folder or its parent folder, the scripts also restart themselves
with that environment when started with any other Python installation.

## Output

`ols_tabelle4.py` prints Table 4. In addition to the table, it prints Holm-adjusted p-values and the
full median regression, which are not reported in the thesis.

`ml_tabelleA2.py` prints Table A2 and saves the panels as CSV files and a text summary in the script
folder (file names `tabelle5_*`, from an earlier table numbering). It also prints a Panel D with the
variable importance for SGrowth, which is not reported in the thesis because no model outperforms the
Dummy Regressor.

Training times in Table A2 depend on the machine and vary between runs. All other values are
reproducible with the package versions listed in `requirements.txt`.

## Data

The workbook contains data from Refinitiv and Bloomberg and is not publicly available due to licensing
restrictions. It is available from the author upon request.
