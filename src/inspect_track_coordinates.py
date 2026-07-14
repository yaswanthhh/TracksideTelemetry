from pathlib import Path
from itertools import combinations

import pandas as pd

INPUT_PATH = Path("data/processed/telemetry_curated.parquet")


def main() -> None:
    telemetry = pd.read_parquet(INPUT_PATH)

    reference_lap_id = (
        telemetry.groupby("lap_id")["lap_time_final_s"]
        .first()
        .sort_values()
        .index[0]
    )

    reference = telemetry.loc[
        telemetry["lap_id"] == reference_lap_id
    ].sort_values("lap_distance")

    axes = ["world_position_X", "world_position_Y", "world_position_Z"]

    print(f"Reference lap: {reference_lap_id}\n")

    for axis in axes:
        values = reference[axis]
        print(
            f"{axis}: "
            f"min={values.min():.3f}, "
            f"max={values.max():.3f}, "
            f"range={values.max() - values.min():.3f}"
        )

    print("\nCoordinate-plane aspect ratios:")

    for x_axis, y_axis in combinations(axes, 2):
        x_range = reference[x_axis].max() - reference[x_axis].min()
        y_range = reference[y_axis].max() - reference[y_axis].min()
        aspect_ratio = max(x_range, y_range) / min(x_range, y_range)

        print(
            f"{x_axis} vs {y_axis}: "
            f"x-range={x_range:.3f}, "
            f"y-range={y_range:.3f}, "
            f"aspect ratio={aspect_ratio:.3f}"
        )


if __name__ == "__main__":
    main()