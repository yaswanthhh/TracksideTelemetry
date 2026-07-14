from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_LAPS_PATH = PROJECT_ROOT / "data/processed/laps_summary.csv"
LOCAL_TELEMETRY_PATH = (
    PROJECT_ROOT / "data/processed/telemetry_curated.parquet"
)

DEMO_LAPS_PATH = PROJECT_ROOT / "data/demo/laps_summary_demo.csv"
DEMO_TELEMETRY_PATH = (
    PROJECT_ROOT / "data/demo/telemetry_curated_demo.parquet"
)

TRACK_ID = "Sachsenring"
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
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if (
        LOCAL_LAPS_PATH.is_file()
        and LOCAL_TELEMETRY_PATH.is_file()
    ):
        laps_path = LOCAL_LAPS_PATH
        telemetry_path = LOCAL_TELEMETRY_PATH
        data_mode = "Full local telemetry dataset"
    elif (
        DEMO_LAPS_PATH.is_file()
        and DEMO_TELEMETRY_PATH.is_file()
    ):
        laps_path = DEMO_LAPS_PATH
        telemetry_path = DEMO_TELEMETRY_PATH
        data_mode = "Public demo dataset: five fastest usable laps"
    else:
        raise FileNotFoundError(
            "No telemetry dataset was found. Expected either local processed "
            "data or the public demo dataset."
        )

    laps = pd.read_csv(laps_path)
    telemetry = pd.read_parquet(telemetry_path)

    usable_laps = laps.loc[laps["is_usable_lap"]].copy()
    usable_laps = usable_laps.sort_values(
        "max_lap_time_s"
    ).reset_index(drop=True)

    telemetry["speed_magnitude"] = np.sqrt(
        telemetry["velocity_X"] ** 2
        + telemetry["velocity_Y"] ** 2
        + telemetry["velocity_Z"] ** 2
    )

    return usable_laps, telemetry, data_mode


def initialise_zone_state() -> None:
    if "zones" not in st.session_state:
        st.session_state.zones = []


def validate_zone(zone: dict) -> dict:
    required_keys = {"name", "start_m", "end_m", "colour"}

    if not isinstance(zone, dict):
        raise ValueError("Each zone must be an object.")

    if set(zone) != required_keys:
        raise ValueError(
            "Each zone must contain only: name, start_m, end_m, colour."
        )

    name = str(zone["name"]).strip()
    start_m = float(zone["start_m"])
    end_m = float(zone["end_m"])
    colour = str(zone["colour"]).strip()

    if not name:
        raise ValueError("Zone names cannot be empty.")

    if start_m < 0 or end_m > TRACK_LENGTH_M or start_m >= end_m:
        raise ValueError(
            f"Zone '{name}' has invalid distance boundaries."
        )

    if not colour.startswith("#") or len(colour) != 7:
        raise ValueError(
            f"Zone '{name}' must use a six-digit hex colour."
        )

    return {
        "name": name,
        "start_m": start_m,
        "end_m": end_m,
        "colour": colour,
    }


def add_zone(
    name: str,
    start_m: float,
    end_m: float,
    colour: str,
) -> None:
    try:
        new_zone = validate_zone(
            {
                "name": name,
                "start_m": start_m,
                "end_m": end_m,
                "colour": colour,
            }
        )
    except ValueError as error:
        st.error(str(error))
        return

    duplicate_name = any(
        zone["name"].lower() == new_zone["name"].lower()
        for zone in st.session_state.zones
    )

    if duplicate_name:
        st.error("Zone names must be unique.")
        return

    st.session_state.zones.append(new_zone)


def build_zone_profile() -> dict:
    return {
        "profile_version": 1,
        "track_id": TRACK_ID,
        "track_length_m": TRACK_LENGTH_M,
        "zones": st.session_state.zones,
    }


def load_zone_profile(uploaded_file) -> None:
    try:
        profile = json.loads(uploaded_file.getvalue().decode("utf-8"))

        if not isinstance(profile, dict):
            raise ValueError("Profile root must be a JSON object.")

        if profile.get("profile_version") != 1:
            raise ValueError("Unsupported or missing profile_version.")

        if profile.get("track_id") != TRACK_ID:
            raise ValueError(
                f"This dashboard requires a {TRACK_ID} zone profile."
            )

        profile_track_length = float(profile.get("track_length_m"))
        if abs(profile_track_length - TRACK_LENGTH_M) > 1.0:
            raise ValueError(
                "The profile track length does not match this circuit."
            )

        zones = profile.get("zones")

        if not isinstance(zones, list):
            raise ValueError("Profile zones must be a list.")

        validated_zones = [validate_zone(zone) for zone in zones]

        zone_names = [zone["name"].lower() for zone in validated_zones]
        if len(zone_names) != len(set(zone_names)):
            raise ValueError("Profile contains duplicate zone names.")

        st.session_state.zones = validated_zones
        st.success(f"Loaded {len(validated_zones)} zone(s) from profile.")

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        st.error(f"Could not load zone profile: {error}")


def filter_to_zone(
    telemetry: pd.DataFrame,
    lap_id: str,
    start_m: float,
    end_m: float,
) -> pd.DataFrame:
    return telemetry.loc[
        (telemetry["lap_id"] == lap_id)
        & telemetry["lap_distance"].between(start_m, end_m)
    ].sort_values("lap_distance")


def zone_metrics(zone_data: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_speed": zone_data["speed_magnitude"].mean(),
        "mean_throttle": zone_data["throttle"].mean(),
        "mean_front_brake": zone_data["brake_0"].mean(),
        "mean_rear_slip": zone_data["wheel_slip_ratio_1"].mean(),
        "rear_slip_event_rate": zone_data["rear_slip_event"].mean(),
    }


def build_track_figure(reference_track: pd.DataFrame) -> go.Figure:
    track = reference_track.copy()

    first_point = track.iloc[[0]].copy()
    first_point["lap_distance"] = TRACK_LENGTH_M

    track = pd.concat(
        [track, first_point],
        ignore_index=True,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=track["world_position_X"],
            y=track["world_position_Y"],
            mode="lines",
            name="Unassigned track",
            line={
                "color": UNASSIGNED_COLOUR,
                "width": 4,
            },
            hoverinfo="skip",
        )
    )

    for zone in st.session_state.zones:
        zone_points = track.loc[
            track["lap_distance"].between(
                zone["start_m"],
                zone["end_m"],
            )
        ]

        if zone_points.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=zone_points["world_position_X"],
                y=zone_points["world_position_Y"],
                mode="lines",
                name=zone["name"],
                line={
                    "color": zone["colour"],
                    "width": 7,
                },
                customdata=zone_points[["lap_distance"]],
                hovertemplate=(
                    f"<b>{zone['name']}</b><br>"
                    "Lap distance: %{customdata[0]:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    figure.update_layout(
        title="Sachsenring reference trajectory and custom zones",
        height=650,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
        legend={
            "orientation": "h",
            "y": -0.12,
        },
    )

    return figure


def build_zone_comparison_figure(
    reference_data: pd.DataFrame,
    selected_data: pd.DataFrame,
    zone_name: str,
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Speed magnitude",
            "Throttle and front brake",
            "Rear wheel-slip ratio",
        ),
    )

    signals = [
        {
            "column": "speed_magnitude",
            "label": "Speed magnitude",
            "row": 1,
            "colour": "#808080",
        },
        {
            "column": "throttle",
            "label": "Throttle",
            "row": 2,
            "colour": "#2E86DE",
        },
        {
            "column": "brake_0",
            "label": "Front brake",
            "row": 2,
            "colour": "#E74C3C",
        },
        {
            "column": "wheel_slip_ratio_1",
            "label": "Rear slip ratio",
            "row": 3,
            "colour": "#8E44AD",
        },
    ]

    for signal in signals:
        figure.add_trace(
            go.Scatter(
                x=reference_data["lap_distance"],
                y=reference_data[signal["column"]],
                mode="lines",
                name=f"Reference — {signal['label']}",
                line={
                    "color": signal["colour"],
                    "width": 2,
                    "dash": "dash",
                },
                legendgroup=f"reference_{signal['column']}",
            ),
            row=signal["row"],
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=selected_data["lap_distance"],
                y=selected_data[signal["column"]],
                mode="lines",
                name=f"Selected — {signal['label']}",
                line={
                    "color": signal["colour"],
                    "width": 3,
                },
                legendgroup=f"selected_{signal['column']}",
            ),
            row=signal["row"],
            col=1,
        )

    figure.update_layout(
        title=f"{zone_name}: selected lap vs reference lap",
        height=850,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": -0.12,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 85,
            "b": 80,
        },
    )

    figure.update_xaxes(
        title_text="Lap distance (m)",
        row=3,
        col=1,
    )

    figure.update_yaxes(
        title_text="Velocity magnitude",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Control input",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Slip ratio",
        row=3,
        col=1,
    )

    return figure


def build_zone_export(
    selected_zone: dict,
    reference_lap_id: str,
    selected_lap_id: str,
    reference_lap_time: float,
    selected_lap_time: float,
    reference_metrics: dict[str, float],
    selected_metrics: dict[str, float],
) -> pd.DataFrame:
    reference_lap_time = round(reference_lap_time, 4)
    selected_lap_time = round(selected_lap_time, 4)

    return pd.DataFrame(
        [
            {
                "zone_name": selected_zone["name"],
                "zone_start_m": selected_zone["start_m"],
                "zone_end_m": selected_zone["end_m"],
                "zone_colour": selected_zone["colour"],
                "reference_lap_id": reference_lap_id,
                "reference_lap_time_s": reference_lap_time,
                "selected_lap_id": selected_lap_id,
                "selected_lap_time_s": selected_lap_time,
                "lap_time_delta_s": (
                    selected_lap_time - reference_lap_time
                ),
                "reference_mean_speed_magnitude": (
                    reference_metrics["mean_speed"]
                ),
                "selected_mean_speed_magnitude": (
                    selected_metrics["mean_speed"]
                ),
                "mean_speed_magnitude_delta": (
                    selected_metrics["mean_speed"]
                    - reference_metrics["mean_speed"]
                ),
                "reference_mean_throttle": (
                    reference_metrics["mean_throttle"]
                ),
                "selected_mean_throttle": (
                    selected_metrics["mean_throttle"]
                ),
                "mean_throttle_delta": (
                    selected_metrics["mean_throttle"]
                    - reference_metrics["mean_throttle"]
                ),
                "reference_mean_front_brake": (
                    reference_metrics["mean_front_brake"]
                ),
                "selected_mean_front_brake": (
                    selected_metrics["mean_front_brake"]
                ),
                "mean_front_brake_delta": (
                    selected_metrics["mean_front_brake"]
                    - reference_metrics["mean_front_brake"]
                ),
                "reference_mean_rear_slip_ratio": (
                    reference_metrics["mean_rear_slip"]
                ),
                "selected_mean_rear_slip_ratio": (
                    selected_metrics["mean_rear_slip"]
                ),
                "mean_rear_slip_ratio_delta": (
                    selected_metrics["mean_rear_slip"]
                    - reference_metrics["mean_rear_slip"]
                ),
                "reference_rear_slip_event_rate": (
                    reference_metrics["rear_slip_event_rate"]
                ),
                "selected_rear_slip_event_rate": (
                    selected_metrics["rear_slip_event_rate"]
                ),
                "rear_slip_event_rate_delta": (
                    selected_metrics["rear_slip_event_rate"]
                    - reference_metrics["rear_slip_event_rate"]
                ),
            }
        ]
    )


def main() -> None:
    initialise_zone_state()

    laps, telemetry, data_mode = load_data()
    st.info(data_mode)

    reference_lap = laps.iloc[0]
    reference_lap_id = reference_lap["lap_id"]
    reference_lap_time = reference_lap["max_lap_time_s"]

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Usable lap segments",
        len(laps),
    )

    metric_2.metric(
        "Reference lap",
        reference_lap_id,
    )

    metric_3.metric(
        "Reference lap time",
        f"{reference_lap_time:.4f} s",
    )

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
            add_zone(
                zone_name,
                start_m,
                end_m,
                zone_colour,
            )

    st.write("Zone profiles")

    profile_column, upload_column = st.columns(2)

    with profile_column:
        profile_json = json.dumps(
            build_zone_profile(),
            indent=2,
        )

        st.download_button(
            label="Download zone profile JSON",
            data=profile_json,
            file_name="sachsenring_zone_profile.json",
            mime="application/json",
            disabled=not st.session_state.zones,
        )

    with upload_column:
        uploaded_profile = st.file_uploader(
            "Upload zone profile JSON",
            type=["json"],
        )

        if uploaded_profile is not None:
            profile_signature = (
                f"{uploaded_profile.name}_"
                f"{uploaded_profile.size}"
            )

            if st.session_state.get(
                "last_loaded_profile"
            ) != profile_signature:
                load_zone_profile(uploaded_profile)
                st.session_state.last_loaded_profile = profile_signature
                st.rerun()

    if st.session_state.zones:
        st.write("Active zones")

        zones_table = pd.DataFrame(
            st.session_state.zones
        ).rename(
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
        st.info(
            "No custom zones yet. Add one above or upload a zone profile."
        )

    st.subheader("Sachsenring track outline")

    reference_track = telemetry.loc[
        telemetry["lap_id"] == reference_lap_id
    ].sort_values("lap_distance")

    track_figure = build_track_figure(reference_track)

    st.plotly_chart(
        track_figure,
        use_container_width=True,
    )

    st.caption(
        "The trajectory is reconstructed from the reference lap's MotoGP 18 "
        "world-position X/Y telemetry. World-position Z is retained as an "
        "elevation-related signal."
    )

    if not st.session_state.zones:
        return

    st.divider()
    st.subheader("Zone performance comparison")

    st.caption(
        "Compares simulator telemetry inside the selected zone. "
        "The reference lap is displayed with dashed traces."
    )

    selected_zone_name = st.selectbox(
        "Analysis zone",
        options=[
            zone["name"]
            for zone in st.session_state.zones
        ],
    )

    selected_zone = next(
        zone
        for zone in st.session_state.zones
        if zone["name"] == selected_zone_name
    )

    lap_options = laps["lap_id"].tolist()

    selected_lap_id = st.selectbox(
        "Selected lap segment",
        options=lap_options,
        index=min(1, len(lap_options) - 1),
    )

    reference_zone_data = filter_to_zone(
        telemetry,
        reference_lap_id,
        selected_zone["start_m"],
        selected_zone["end_m"],
    )

    selected_zone_data = filter_to_zone(
        telemetry,
        selected_lap_id,
        selected_zone["start_m"],
        selected_zone["end_m"],
    )

    if reference_zone_data.empty or selected_zone_data.empty:
        st.error(
            "No telemetry data is available for this selected zone."
        )
        return

    reference_metrics = zone_metrics(reference_zone_data)
    selected_metrics = zone_metrics(selected_zone_data)

    selected_lap_time = laps.loc[
        laps["lap_id"] == selected_lap_id,
        "max_lap_time_s",
    ].iloc[0]

    time_delta = selected_lap_time - reference_lap_time

    st.write(
        f"Selected lap: **{selected_lap_id}** "
        f"({selected_lap_time:.4f} s, "
        f"{time_delta:+.4f} s vs reference)"
    )

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Mean speed magnitude",
        f"{selected_metrics['mean_speed']:.2f}",
        (
            f"{selected_metrics['mean_speed'] - reference_metrics['mean_speed']:+.2f}"
        ),
    )

    metric_columns[1].metric(
        "Mean throttle",
        f"{selected_metrics['mean_throttle']:.3f}",
        (
            f"{selected_metrics['mean_throttle'] - reference_metrics['mean_throttle']:+.3f}"
        ),
    )

    metric_columns[2].metric(
        "Mean front brake",
        f"{selected_metrics['mean_front_brake']:.3f}",
        (
            f"{selected_metrics['mean_front_brake'] - reference_metrics['mean_front_brake']:+.3f}"
        ),
    )

    metric_columns[3].metric(
        "Mean rear-slip ratio",
        f"{selected_metrics['mean_rear_slip']:.3f}",
        (
            f"{selected_metrics['mean_rear_slip'] - reference_metrics['mean_rear_slip']:+.3f}"
        ),
    )

    metric_columns[4].metric(
        "Rear-slip event rate",
        f"{selected_metrics['rear_slip_event_rate'] * 100:.1f}%",
        (
            f"{(selected_metrics['rear_slip_event_rate'] - reference_metrics['rear_slip_event_rate']) * 100:+.1f}%"
        ),
    )

    comparison_figure = build_zone_comparison_figure(
        reference_zone_data,
        selected_zone_data,
        selected_zone["name"],
    )

    st.plotly_chart(
        comparison_figure,
        use_container_width=True,
    )

    export_data = build_zone_export(
        selected_zone,
        reference_lap_id,
        selected_lap_id,
        reference_lap_time,
        selected_lap_time,
        reference_metrics,
        selected_metrics,
    ).round(6)

    safe_zone_name = (
        selected_zone["name"]
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    st.download_button(
        label="Download selected-zone comparison CSV",
        data=export_data.to_csv(index=False).encode("utf-8"),
        file_name=f"zone_comparison_{safe_zone_name}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()