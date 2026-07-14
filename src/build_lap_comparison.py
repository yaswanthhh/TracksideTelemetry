from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/processed/track_zone_metrics.parquet")
OUTPUT_PATH = Path("data/processed/lap_comparison_metrics.parquet")


def main() -> None:
    zone_metrics = pd.read_parquet(INPUT_PATH)

    best_lap_id = (
        zone_metrics.sort_values("lap_time_final_s")
        .iloc[0]["lap_id"]
    )

    best_lap_time_s = (
        zone_metrics.loc[
            zone_metrics["lap_id"] == best_lap_id,
            "lap_time_final_s",
        ]
        .iloc[0]
    )

    reference = (
        zone_metrics.loc[zone_metrics["lap_id"] == best_lap_id]
        .loc[
            :,
            [
                "track_zone",
                "mean_throttle",
                "mean_front_brake",
                "mean_brake_total",
                "mean_rpm",
                "mean_wheel_speed_delta",
                "max_rear_slip_ratio",
                "mean_rear_slip_ratio",
                "rear_slip_event_rate",
                "mean_throttle_rate",
                "mean_gforce_x",
                "mean_gforce_y",
            ],
        ]
        .rename(
            columns={
                "mean_throttle": "reference_mean_throttle",
                "mean_front_brake": "reference_mean_front_brake",
                "mean_brake_total": "reference_mean_brake_total",
                "mean_rpm": "reference_mean_rpm",
                "mean_wheel_speed_delta": "reference_wheel_speed_delta",
                "max_rear_slip_ratio": "reference_max_rear_slip_ratio",
                "mean_rear_slip_ratio": "reference_mean_rear_slip_ratio",
                "rear_slip_event_rate": "reference_rear_slip_event_rate",
                "mean_throttle_rate": "reference_mean_throttle_rate",
                "mean_gforce_x": "reference_mean_gforce_x",
                "mean_gforce_y": "reference_mean_gforce_y",
            }
        )
    )

    comparison = zone_metrics.merge(
        reference,
        on="track_zone",
        how="left",
        validate="many_to_one",
    )

    comparison["lap_time_delta_s"] = (
        comparison["lap_time_final_s"] - best_lap_time_s
    )

    comparison["throttle_delta_vs_reference"] = (
        comparison["mean_throttle"]
        - comparison["reference_mean_throttle"]
    )

    comparison["front_brake_delta_vs_reference"] = (
        comparison["mean_front_brake"]
        - comparison["reference_mean_front_brake"]
    )

    comparison["rear_slip_delta_vs_reference"] = (
        comparison["mean_rear_slip_ratio"]
        - comparison["reference_mean_rear_slip_ratio"]
    )

    comparison["rear_slip_event_rate_delta_vs_reference"] = (
        comparison["rear_slip_event_rate"]
        - comparison["reference_rear_slip_event_rate"]
    )

    comparison = comparison.sort_values(
        ["lap_time_final_s", "track_zone"]
    ).reset_index(drop=True)

    comparison.to_parquet(OUTPUT_PATH, index=False)

    print(f"Reference lap: {best_lap_id}")
    print(f"Reference lap time: {best_lap_time_s:.4f} s")
    print(f"Laps compared: {comparison['lap_id'].nunique()}")
    print(f"Lap-zone comparisons: {len(comparison):,}")
    print(f"Saved: {OUTPUT_PATH}\n")

    display_columns = [
        "lap_id",
        "lap_time_final_s",
        "lap_time_delta_s",
        "track_zone_label",
        "throttle_delta_vs_reference",
        "front_brake_delta_vs_reference",
        "rear_slip_delta_vs_reference",
    ]

    print(comparison[display_columns].head(12).to_string(index=False))


if __name__ == "__main__":
    main()