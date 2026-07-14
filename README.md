# TracksideTelemetry
##[Live Demo](https://tracksidetelemetry.streamlit.app)
A Streamlit dashboard for analysing MotoGP 18 telemetry data. It provides lap viewing, lap comparison, editable track zones, telemetry-based findings, and CSV exports.

> This is a simulator-data analysis project. It does not control a motorcycle ECU and does not provide real-world riding advice.

## Features

- Lap leaderboard with best-lap and coverage metrics
- Track trajectory viewer for each available lap
- Starter and custom track-analysis zones
- Zone profile export/import as JSON
- Reference-versus-comparison lap analysis
- Speed, throttle, front-brake, and rear-slip comparison charts
- Ranked zone findings and selected-zone CSV export

## Demo data

The deployed version uses a lightweight public demo dataset with 5 usable laps:

```text
data/demo/laps_summary_demo.csv
data/demo/telemetry_curated_demo.parquet
```

When full local files are available, the dashboard loads them first:

```text
data/processed/laps_summary.csv
data/processed/telemetry_curated.parquet
```

## Requirements

- Python 3.10 or newer
- pip

## Run locally

1. Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/TracksideTelemetry.git
cd TracksideTelemetry
```

2. Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```bat
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install streamlit pandas numpy plotly pyarrow
```

4. Start the dashboard:

```bash
streamlit run app/dashboard.py
```

5. Open the local address shown in the terminal, usually:

```text
http://localhost:8501
```

## How to use

1. Open the **Dashboard** tab to view the lap leaderboard and track trajectory.
2. Select a lap from **Choose a lap to display**.
3. Open **Zone manager** and click **Load starter zones**, or create custom zones.
4. Open **Lap comparison**, select separate reference and comparison laps, then review zone findings.
5. Select a zone to inspect speed, throttle, braking, and rear-slip traces.
6. Download the selected-zone comparison as a CSV if required.

## Project structure

```text
TracksideTelemetry/
├── app/
│   └── dashboard.py
├── data/
│   ├── demo/
│   │   ├── laps_summary_demo.csv
│   │   └── telemetry_curated_demo.parquet
│   └── processed/             # Optional full local dataset
│       ├── laps_summary.csv
│       └── telemetry_curated.parquet
└── README.md
```

## Data priority

The app automatically uses:

1. `data/processed/` when the full local dataset is available
2. `data/demo/` when the full dataset is unavailable

The active source is displayed in the dashboard as either **Full local telemetry dataset** or **Public demo dataset**.
