from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

LAPS_PATH = PROJECT_ROOT / "data/processed/laps_summary.csv"
TELEMETRY_PATH = PROJECT_ROOT / "data/processed/telemetry_curated.parquet"

st.set_page_config(
    page_title="TracksideTelemetry",
    page_icon="🏍️",
    layout="wide",
)

st.title("🏍️ TracksideTelemetry")
st.caption(
    "MotoGP 18 telemetry analysis and simulated calibration-workflow prototype"
)

st.warning(
    "Simulator-data research project only. "
    "This app does not control a motorcycle ECU and does not provide "
    "real-world racing or safety recommendations."
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    laps = pd.read_csv(LAPS_PATH)
    telemetry = pd.read_parquet(TELEMETRY_PATH)

    usable_laps = laps.loc[laps["is_usable_lap"]].copy()
    usable_laps = usable_laps.sort_values("max_lap_time_s").reset_index(drop=True)

    return usable_laps, telemetry


def main() -> None:
    laps, telemetry = load_data()

    reference_lap = laps.iloc[0]
    reference_lap_id = reference_lap["lap_id"]
    reference_lap_time = reference_lap["max_lap_time_s"]

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Usable lap segments", len(laps))
    metric_2.metric("Reference lap", reference_lap_id)
    metric_3.metric("Reference lap time", f"{reference_lap_time:.4f} s")

    st.subheader("Lap-time leaderboard")

    leaderboard = laps.loc[
        :,
        [
            "lap_id",
            "max_lap_time_s",
            "distance_coverage_pct",
            "sample_count",
            "max_rear_slip",
        ],
    ].rename(
        columns={
            "lap_id": "Lap segment",
            "max_lap_time_s": "Lap time (s)",
            "distance_coverage_pct": "Distance coverage (%)",
            "sample_count": "Samples",
            "max_rear_slip": "Max rear slip ratio",
        }
    )

    st.dataframe(
        leaderboard,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Sachsenring track outline")

    reference_track = telemetry.loc[
        telemetry["lap_id"] == reference_lap_id
    ].sort_values("lap_distance").copy()

    track_for_plot = pd.concat(
        [reference_track, reference_track.iloc[[0]]],
        ignore_index=True,
    )

    track_figure = px.line(
        track_for_plot,
        x="world_position_X",
        y="world_position_Y",
        title="Sachsenring reference-lap trajectory",
        labels={
            "world_position_X": "World position X",
        "world_position_Y": "World position Y",
    },
)

    track_figure.update_traces(line={"color": "#808080", "width": 3})
    track_figure.update_yaxes(scaleanchor="x", scaleratio=1)
    track_figure.update_layout(
        height=650,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )

    st.plotly_chart(track_figure, use_container_width=True)

    st.caption(
        "The trajectory is reconstructed from the reference lap's "
        "MotoGP 18 world-position X/Y telemetry. "
        "World-position Z is retained as an elevation-related signal."
    )


if __name__ == "__main__":
    main()