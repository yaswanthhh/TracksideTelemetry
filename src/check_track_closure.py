from pathlib import Path

import numpy as np
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

    track = (
        telemetry.loc[telemetry["lap_id"] == reference_lap_id]
        .sort_values("lap_distance")
        .reset_index(drop=True)
    )

    start = track.iloc[0]
    end = track.iloc[-1]

    closure_distance = np.hypot(
        end["world_position_X"] - start["world_position_X"],
        end["world_position_Y"] - start["world_position_Y"],
    )

    print(f"Reference lap: {reference_lap_id}")
    print(f"Samples: {len(track):,}")
    print(f"Start distance: {start['lap_distance']:.2f} m")
    print(f"End distance: {end['lap_distance']:.2f} m")
    print(f"Track length: {track['trackLength'].iloc[0]:.2f} m")
    print(f"Start/end closure gap: {closure_distance:.3f} m")


if __name__ == "__main__":
    main()