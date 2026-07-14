from pathlib import Path

import pandas as pd

LAPS_PATH = Path("data/processed/laps_summary.csv")
TELEMETRY_PATH = Path("data/processed/telemetry_curated.parquet")
DEMO_DIR = Path("data/demo")

DEMO_LAP_COUNT = 5


def main() -> None:
    laps = pd.read_csv(LAPS_PATH)
    telemetry = pd.read_parquet(TELEMETRY_PATH)

    demo_laps = (
        laps.loc[laps["is_usable_lap"]]
        .head(DEMO_LAP_COUNT)
        .copy()
    )

    demo_lap_ids = demo_laps["lap_id"].tolist()

    demo_telemetry = (
        telemetry.loc[telemetry["lap_id"].isin(demo_lap_ids)]
        .sort_values(["lap_time_final_s", "lap_distance"])
        .copy()
    )

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    laps_output = DEMO_DIR / "laps_summary_demo.csv"
    telemetry_output = DEMO_DIR / "telemetry_curated_demo.parquet"

    demo_laps.to_csv(laps_output, index=False)
    demo_telemetry.to_parquet(telemetry_output, index=False)

    print(f"Demo lap segments: {len(demo_laps)}")
    print(f"Demo telemetry rows: {len(demo_telemetry):,}")
    print(
        "Lap-time range: "
        f"{demo_laps['max_lap_time_s'].min():.4f}–"
        f"{demo_laps['max_lap_time_s'].max():.4f} s"
    )
    print(f"Saved: {laps_output}")
    print(f"Saved: {telemetry_output}")


if __name__ == "__main__":
    main()