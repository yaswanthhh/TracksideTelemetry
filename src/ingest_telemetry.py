from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

REQUIRED_COLUMNS = {
    "carId",
    "trackId",
    "trackLength",
    "lapIndex",
    "lapNum",
    "lapFlag",
    "binIndex",
    "validBin",
    "lap_number",
    "lap_distance",
    "lap_time",
    "lap_time_invalid",
    "throttle",
    "brake_0",
    "brake_1",
    "steering",
    "gear",
    "rpm",
    "wheel_speed_0",
    "wheel_speed_1",
    "wheel_slip_ratio_0",
    "wheel_slip_ratio_1",
    "susp_pos_0",
    "susp_pos_1",
}

NUMERIC_COLUMNS = [
    "trackLength",
    "lapIndex",
    "lapNum",
    "lapFlag",
    "binIndex",
    "validBin",
    "lap_number",
    "lap_distance",
    "lap_time",
    "lap_time_invalid",
    "throttle",
    "brake_0",
    "brake_1",
    "steering",
    "gear",
    "rpm",
    "wheel_speed_0",
    "wheel_speed_1",
    "wheel_slip_ratio_0",
    "wheel_slip_ratio_1",
    "susp_pos_0",
    "susp_pos_1",
]


def read_telemetry_file(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        csv_path,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False,
    )

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing: {sorted(missing)}")

    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["source_file"] = csv_path.name
    return frame


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found in data/raw.")

    frames = [read_telemetry_file(csv_path) for csv_path in files]
    telemetry = pd.concat(frames, ignore_index=True)

    telemetry["lap_id"] = (
        telemetry["source_file"]
        .str.replace(".csv", "", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(",", "_", regex=False)
        + "_segment_"
        + telemetry["lapIndex"].fillna(-1).astype(int).astype(str)
    )

    telemetry = telemetry.sort_values(
        ["source_file", "lapIndex", "binIndex"]
    ).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "telemetry_points.parquet"
    telemetry.to_parquet(output_path, index=False)

    print(f"Files ingested: {len(files)}")
    print(f"Telemetry rows: {len(telemetry):,}")
    print(f"Columns: {len(telemetry.columns)}")
    print(f"Recorded lap segments: {telemetry['lap_id'].nunique():,}")
    print(f"Tracks: {', '.join(sorted(telemetry['trackId'].dropna().unique()))}")
    print(f"Bikes: {', '.join(sorted(telemetry['carId'].dropna().unique()))}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()