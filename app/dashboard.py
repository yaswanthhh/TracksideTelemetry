from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

LAPS_PATH = PROJECT_ROOT / "data/processed/laps_summary.csv"
TELEMETRY_PATH = PROJECT_ROOT / "data/processed/telemetry_curated.parquet"

TRACK_LENGTH_M = 3667.63
DEFAULT_ZONE_COLOUR = "#E74C3C"
UNASSIGNED_COLOUR = "#808080"

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


def initialise_zone_state() -> None:
    if "zones" not in st.session_state:
        st.session_state.zones = []


def add_zone(name: str, start_m: float, end_m: float, colour: str) -> None:
    if start_m >= end_m:
        st.error("Zone end distance must be greater than its start distance.")
        return

    duplicate_name = any(
        zone["name"].lower() == name.strip().lower()
        for zone in st.session_state.zones
    )

    if duplicate_name:
        st.error("Zone names must be unique.")
        return

    st.session_state.zones.append(
        {
            "name": name.strip(),
            "start_m": float(start_m),
            "end_m": float(end_m),
            "colour": colour,
        }
    )


def build_track_figure(reference_track: pd.DataFrame) -> go.Figure:
    track = reference_track.copy()

    first_point = track.iloc[[0]].copy()
    first_point["lap_distance"] = TRACK_LENGTH_M
    track = pd.concat([track, first_point], ignore_index=True)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=track["world_position_X"],
            y=track["world_position_Y"],
            mode="lines",
            name="Unassigned track",
            line={"color": UNASSIGNED_COLOUR, "width": 4},
            hoverinfo="skip",
        )
    )

    for zone in st.session_state.zones:
        zone_points = track.loc[
            (track["lap_distance"] >= zone["start_m"])
            & (track["lap_distance"] <= zone["end_m"])
        ]

        if zone_points.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=zone_points["world_position_X"],
                y=zone_points["world_position_Y"],
                mode="lines",
                name=zone["name"],
                line={"color": zone["colour"], "width": 7},
                customdata=zone_points[["lap_distance"]],
                hovertemplate=(
                    f"<b>{zone['name']}</b><br>"
                    "Lap distance: %{customdata[0]:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figure.update_layout(
        title="Sachsenring reference trajectory and custom zones",
        height=650,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "y": -0.12},
    )

    return figure


def main() -> None:
    initialise_zone_state()
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

    st.subheader("Track-zone manager")
    st.caption(
        "Define simulator-analysis zones by lap distance. "
        "Zones are visual overlays only and do not modify telemetry."
    )

    with st.form("add_zone_form", clear_on_submit=True):
        name_column, start_column, end_column, colour_column = st.columns(4)

        with name_column:
            zone_name = st.text_input(
                "Zone name",
                placeholder="e.g. Turn 1 braking",
            )

        with start_column:
            start_m = st.number_input(
                "Start distance (m)",
                min_value=0.0,
                max_value=TRACK_LENGTH_M,
                value=0.0,
                step=10.0,
            )

        with end_column:
            end_m = st.number_input(
                "End distance (m)",
                min_value=0.0,
                max_value=TRACK_LENGTH_M,
                value=100.0,
                step=10.0,
            )

        with colour_column:
            zone_colour = st.color_picker(
                "Zone colour",
                value=DEFAULT_ZONE_COLOUR,
            )

        submitted = st.form_submit_button("Add zone")

    if submitted:
        if not zone_name.strip():
            st.error("Enter a zone name.")
        else:
            add_zone(zone_name, start_m, end_m, zone_colour)

    if st.session_state.zones:
        st.write("Active zones")

        zones_table = pd.DataFrame(st.session_state.zones).rename(
            columns={
                "name": "Zone",
                "start_m": "Start (m)",
                "end_m": "End (m)",
                "colour": "Colour",
            }
        )

        st.dataframe(
            zones_table,
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear all zones"):
            st.session_state.zones = []
            st.rerun()
    else:
        st.info("No custom zones yet. Add one above to colour the track.")

    st.subheader("Sachsenring track outline")

    reference_track = telemetry.loc[
        telemetry["lap_id"] == reference_lap_id
    ].sort_values("lap_distance")

    track_figure = build_track_figure(reference_track)
    st.plotly_chart(track_figure, use_container_width=True)

    st.caption(
        "The trajectory is reconstructed from the reference lap's MotoGP 18 "
        "world-position X/Y telemetry. World-position Z is retained as an "
        "elevation-related signal."
    )


if __name__ == "__main__":
    main()