from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/telemetry_points.parquet")
OUTPUT_PATH = Path("data/processed/laps_summary.csv")


def main() -> None:
    telemetry = pd.read_parquet(INPUT_PATH)

    summary = (
        telemetry.groupby("lap_id", as_index=False)
        .agg(
            source_file=("source_file", "first"),
            track_id=("trackId", "first"),
            bike_id=("carId", "first"),
            lap_index=("lapIndex", "first"),
            lap_number=("lap_number", "first"),
            sample_count=("binIndex", "size"),
            max_distance_m=("lap_distance", "max"),
            max_lap_time_s=("lap_time", "max"),
            invalid_flag_max=("lap_time_invalid", "max"),
            valid_bin_rate=("validBin", "mean"),
            max_throttle=("throttle", "max"),
            max_front_brake=("brake_0", "max"),
            max_rear_brake=("brake_1", "max"),
            max_rpm=("rpm", "max"),
            max_rear_slip=("wheel_slip_ratio_1", "max"),
            mean_rear_slip=("wheel_slip_ratio_1", "mean"),
            max_front_brake_temp=("brake_temp_0", "max"),
            max_rear_brake_temp=("brake_temp_1", "max"),
            max_front_tyre_wear=("tyre_wear_0", "max"),
            max_rear_tyre_wear=("tyre_wear_1", "max"),
        )
    )

    track_length = telemetry["trackLength"].median()

    summary["distance_coverage_pct"] = (
        summary["max_distance_m"] / track_length * 100
    )

    summary["is_complete_lap"] = summary["distance_coverage_pct"] >= 95
    summary["is_valid_lap"] = summary["invalid_flag_max"].fillna(1).eq(0)

    summary["is_usable_lap"] = (
        summary["is_complete_lap"]
        & summary["is_valid_lap"]
        & (summary["sample_count"] >= 100)
        & (summary["max_lap_time_s"] > 30)
    )

    summary = summary.sort_values(
        ["is_usable_lap", "max_lap_time_s"],
        ascending=[False, True],
    ).reset_index(drop=True)

    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Total lap records: {len(summary)}")
    print(f"Complete laps: {summary['is_complete_lap'].sum()}")
    print(f"Valid laps: {summary['is_valid_lap'].sum()}")
    print(f"Usable laps: {summary['is_usable_lap'].sum()}")
    print(f"Track length: {track_length:.2f} m")
    print(f"Saved: {OUTPUT_PATH}\n")

    display_columns = [
        "lap_id",
        "lap_number",
        "sample_count",
        "max_distance_m",
        "max_lap_time_s",
        "invalid_flag_max",
        "distance_coverage_pct",
        "is_usable_lap",
    ]

    print(summary[display_columns].head(15).to_string(index=False))


if __name__ == "__main__":
    main()