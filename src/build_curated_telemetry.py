from pathlib import Path

import numpy as np
import pandas as pd

TELEMETRY_PATH = Path("data/processed/telemetry_points.parquet")
LAPS_PATH = Path("data/processed/laps_summary.csv")
OUTPUT_PATH = Path("data/processed/telemetry_curated.parquet")

TRACK_ZONE_SIZE_M = 100
REAR_SLIP_EVENT_THRESHOLD = 0.10


def main() -> None:
    telemetry = pd.read_parquet(TELEMETRY_PATH)
    laps = pd.read_csv(LAPS_PATH)

    usable_laps = laps.loc[laps["is_usable_lap"], [
        "lap_id",
        "max_lap_time_s",
        "distance_coverage_pct",
    ]].copy()

    usable_laps = usable_laps.rename(
        columns={"max_lap_time_s": "lap_time_final_s"}
    )

    curated = telemetry.merge(
        usable_laps,
        on="lap_id",
        how="inner",
        validate="many_to_one",
    )

    curated["track_zone"] = (
        np.floor(curated["lap_distance"] / TRACK_ZONE_SIZE_M)
        .astype("Int64")
        * TRACK_ZONE_SIZE_M
    )

    curated["track_zone_label"] = (
        curated["track_zone"].astype(str)
        + "-"
        + (curated["track_zone"] + TRACK_ZONE_SIZE_M).astype(str)
        + " m"
    )

    curated["wheel_speed_delta"] = (
        curated["wheel_speed_1"] - curated["wheel_speed_0"]
    )

    curated["brake_total"] = curated["brake_0"] + curated["brake_1"]

    curated["rear_slip_event"] = (
        curated["wheel_slip_ratio_1"] >= REAR_SLIP_EVENT_THRESHOLD
    )

    curated["throttle_rate_per_sample"] = (
        curated.groupby("lap_id")["throttle"]
        .diff()
        .fillna(0)
    )

    curated = curated.sort_values(
        ["lap_time_final_s", "lap_id", "binIndex"]
    ).reset_index(drop=True)

    curated.to_parquet(OUTPUT_PATH, index=False)

    print(f"Usable laps included: {curated['lap_id'].nunique()}")
    print(f"Curated telemetry rows: {len(curated):,}")
    print(f"Track-zone size: {TRACK_ZONE_SIZE_M} m")
    print(
        "Rear-slip events: "
        f"{curated['rear_slip_event'].sum():,} "
        f"({curated['rear_slip_event'].mean() * 100:.2f}% of samples)"
    )
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
