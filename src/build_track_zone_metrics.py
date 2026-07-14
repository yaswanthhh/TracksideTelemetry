from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/telemetry_curated.parquet")
OUTPUT_PATH = Path("data/processed/track_zone_metrics.parquet")


def main() -> None:
    telemetry = pd.read_parquet(INPUT_PATH)

    zone_metrics = (
        telemetry.groupby(
            ["lap_id", "lap_time_final_s", "track_zone", "track_zone_label"],
            as_index=False,
        )
        .agg(
            sample_count=("binIndex", "size"),
            mean_throttle=("throttle", "mean"),
            max_throttle=("throttle", "max"),
            mean_front_brake=("brake_0", "mean"),
            max_front_brake=("brake_0", "max"),
            mean_rear_brake=("brake_1", "mean"),
            mean_brake_total=("brake_total", "mean"),
            mean_rpm=("rpm", "mean"),
            mean_wheel_speed_delta=("wheel_speed_delta", "mean"),
            max_rear_slip_ratio=("wheel_slip_ratio_1", "max"),
            mean_rear_slip_ratio=("wheel_slip_ratio_1", "mean"),
            rear_slip_event_count=("rear_slip_event", "sum"),
            rear_slip_event_rate=("rear_slip_event", "mean"),
            mean_throttle_rate=("throttle_rate_per_sample", "mean"),
            mean_front_suspension=("susp_pos_0", "mean"),
            mean_rear_suspension=("susp_pos_1", "mean"),
            mean_gforce_x=("gforce_X", "mean"),
            mean_gforce_y=("gforce_Y", "mean"),
        )
    )

    zone_metrics["rear_slip_event_count"] = (
        zone_metrics["rear_slip_event_count"].astype(int)
    )

    zone_metrics = zone_metrics.sort_values(
        ["lap_time_final_s", "track_zone"]
    ).reset_index(drop=True)

    zone_metrics.to_parquet(OUTPUT_PATH, index=False)

    print(f"Usable laps analysed: {zone_metrics['lap_id'].nunique()}")
    print(f"Track zones per lap: {zone_metrics['track_zone'].nunique()}")
    print(f"Lap-zone records: {len(zone_metrics):,}")
    print(f"Saved: {OUTPUT_PATH}\n")

    display_columns = [
        "lap_id",
        "lap_time_final_s",
        "track_zone_label",
        "mean_throttle",
        "mean_front_brake",
        "max_rear_slip_ratio",
        "rear_slip_event_rate",
    ]

    print(zone_metrics[display_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()