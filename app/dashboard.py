from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_LAPS_PATH = PROJECT_ROOT / "data" / "processed" / "laps_summary.csv"
LOCAL_TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telemetry_curated.parquet"
)

DEMO_LAPS_PATH = PROJECT_ROOT / "data" / "demo" / "laps_summary_demo.csv"
DEMO_TELEMETRY_PATH = (
    PROJECT_ROOT / "data" / "demo" / "telemetry_curated_demo.parquet"
)

DEFAULT_TRACK_NAME = "Sachsenring"
DEFAULT_TRACK_LENGTH_M = 3667.63
DEFAULT_ZONE_COLOUR = "#E74C3C"
UNASSIGNED_COLOUR = "#6B7280"

STARTER_ZONE_RATIOS = [
    ("Start / finish reference", 0.00, 0.05, "#6B7280"),
    ("Early braking analysis", 0.05, 0.10, "#E74C3C"),
    ("Early drive analysis", 0.10, 0.23, "#2E86DE"),
    ("Mid-lap control", 0.23, 0.41, "#8E44AD"),
    ("Technical transition", 0.41, 0.60, "#F39C12"),
    ("Late braking analysis", 0.60, 0.76, "#C0392B"),
    ("Final drive analysis", 0.76, 1.00, "#27AE60"),
]

st.set_page_config(
    page_title="TracksideTelemetry",
    page_icon="🏍️",
    layout="wide",
)

def normalise_telemetry_columns(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    telemetry = telemetry.copy()

    column_mapping = {
        "lapIndex": "lap_index",
        "lapdistance": "lap_distance",
        "laptime": "lap_time",
        "worldpositionX": "world_position_X",
        "worldpositionY": "world_position_Y",
        "worldpositionZ": "world_position_Z",
        "velocityX": "velocity_X",
        "velocityY": "velocity_Y",
        "velocityZ": "velocity_Z",
        "brake0": "brake_0",
        "brake1": "brake_1",
        "wheelslipratio1": "wheel_slip_ratio_1",
    }

    rename_columns = {
        original: renamed
        for original, renamed in column_mapping.items()
        if original in telemetry.columns
        and renamed not in telemetry.columns
    }

    return telemetry.rename(columns=rename_columns)
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if LOCAL_LAPS_PATH.is_file() and LOCAL_TELEMETRY_PATH.is_file():
        laps_path = LOCAL_LAPS_PATH
        telemetry_path = LOCAL_TELEMETRY_PATH
        data_mode = "Full local telemetry dataset"

    elif DEMO_LAPS_PATH.is_file() and DEMO_TELEMETRY_PATH.is_file():
        laps_path = DEMO_LAPS_PATH
        telemetry_path = DEMO_TELEMETRY_PATH
        data_mode = "Public demo dataset"

    else:
        raise FileNotFoundError(
            "No telemetry data found. Expected processed data or demo data."
        )

    laps = pd.read_csv(laps_path)
    telemetry = normalise_telemetry_columns(
        pd.read_parquet(telemetry_path)
    )

    laps = laps.copy()
    telemetry = telemetry.copy()

    if "is_usable_lap" in laps.columns:
        laps = laps.loc[laps["is_usable_lap"]].copy()

    telemetry["speed_magnitude"] = np.sqrt(
        telemetry["velocity_X"] ** 2
        + telemetry["velocity_Y"] ** 2
        + telemetry["velocity_Z"] ** 2
    )

    # If the summary does not already retain a chronological lap index,
    # recover it from the telemetry data using lap_id.
    lap_index_columns = [
        "lap_index",
        "lapIndex",
        "lap_number",
        "lapnumber",
        "lapNum",
    ]

    summary_has_order_column = any(
        column in laps.columns
        for column in lap_index_columns
    )

    telemetry_order_column = next(
        (
            column
            for column in [
                "lap_index",
                "lapIndex",
                "lap_number",
                "lapnumber",
                "lapNum",
            ]
            if column in telemetry.columns
        ),
        None,
    )

    if (
        not summary_has_order_column
        and telemetry_order_column is not None
        and "lap_id" in laps.columns
        and "lap_id" in telemetry.columns
    ):
        lap_order = (
            telemetry.groupby("lap_id", as_index=False)
            .agg(
                lapIndex=(
                    telemetry_order_column,
                    "min",
                )
            )
        )

        laps = laps.merge(
            lap_order,
            on="lap_id",
            how="left",
            validate="one_to_one",
        )

    return laps, telemetry, data_mode


def get_track_length(telemetry: pd.DataFrame) -> float:
    if "lap_distance" not in telemetry.columns:
        return DEFAULT_TRACK_LENGTH_M

    measured_length = float(telemetry["lap_distance"].max())

    if measured_length <= 0:
        return DEFAULT_TRACK_LENGTH_M

    return measured_length


def initialise_state() -> None:
    if "zones" not in st.session_state:
        st.session_state.zones = []

    if "loaded_profile_signature" not in st.session_state:
        st.session_state.loaded_profile_signature = None


def create_lap_labels(laps: pd.DataFrame) -> pd.DataFrame:
    clean_laps = laps.copy()

    # Always use the actual chronological lap index when it exists.
    # Do NOT use _input_order first: the processed CSV may already be
    # written in fastest-to-slowest order.
    chronological_column = next(
        (
            column
            for column in [
                "lap_index",
                "lapIndex",
                "lap_number",
                "lapnumber",
                "lapNum",
            ]
            if column in clean_laps.columns
        ),
        None,
    )

    if chronological_column is not None:
        clean_laps[chronological_column] = pd.to_numeric(
            clean_laps[chronological_column],
            errors="coerce",
        )

        clean_laps = (
            clean_laps
            .sort_values(
                chronological_column,
                ascending=True,
                kind="stable",
                na_position="last",
            )
            .reset_index(drop=True)
        )

    elif "_input_order" in clean_laps.columns:
        # Only use original CSV order when no lap-index field exists.
        clean_laps = (
            clean_laps
            .sort_values("_input_order", kind="stable")
            .reset_index(drop=True)
        )

    else:
        clean_laps = clean_laps.reset_index(drop=True)

    # Display number follows chronological order in the selector.
    clean_laps["display_lap_number"] = range(1, len(clean_laps) + 1)

    best_lap_time = float(clean_laps["max_lap_time_s"].min())

    def label_row(row: pd.Series) -> str:
        lap_number = int(row["display_lap_number"])
        lap_time = float(row["max_lap_time_s"])
        delta = lap_time - best_lap_time

        if abs(delta) < 0.0005:
            return f"Lap {lap_number} — {lap_time:.3f} s — Best lap"

        return (
            f"Lap {lap_number} — {lap_time:.3f} s "
            f"({delta:+.3f} s)"
        )

    clean_laps["lap_label"] = clean_laps.apply(label_row, axis=1)

    return clean_laps


def build_starter_zones(track_length_m: float) -> list[dict]:
    zones = []

    for name, start_ratio, end_ratio, colour in STARTER_ZONE_RATIOS:
        zones.append(
            {
                "name": name,
                "start_m": round(start_ratio * track_length_m, 1),
                "end_m": round(end_ratio * track_length_m, 1),
                "colour": colour,
            }
        )

    return zones


def validate_zone(zone: dict, track_length_m: float) -> dict:
    required_keys = {"name", "start_m", "end_m", "colour"}

    if not isinstance(zone, dict):
        raise ValueError("Each zone must be a JSON object.")

    if set(zone.keys()) != required_keys:
        raise ValueError(
            "Each zone must contain name, start_m, end_m, and colour."
        )

    name = str(zone["name"]).strip()
    start_m = float(zone["start_m"])
    end_m = float(zone["end_m"])
    colour = str(zone["colour"]).strip()

    if not name:
        raise ValueError("Zone names cannot be empty.")

    if start_m < 0 or end_m > track_length_m or start_m >= end_m:
        raise ValueError(
            f"Zone '{name}' has invalid start or end distances."
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
    track_length_m: float,
) -> None:
    try:
        new_zone = validate_zone(
            {
                "name": name,
                "start_m": start_m,
                "end_m": end_m,
                "colour": colour,
            },
            track_length_m,
        )
    except ValueError as error:
        st.error(str(error))
        return

    name_exists = any(
        zone["name"].lower() == new_zone["name"].lower()
        for zone in st.session_state.zones
    )

    if name_exists:
        st.error("Zone names must be unique.")
        return

    st.session_state.zones.append(new_zone)
    st.success(f"Added zone: {new_zone['name']}")


def build_zone_profile(
    track_name: str,
    track_length_m: float,
) -> dict:
    return {
        "profile_version": 1,
        "track_name": track_name,
        "track_length_m": track_length_m,
        "zones": st.session_state.zones,
    }


def load_zone_profile(
    uploaded_file,
    track_name: str,
    track_length_m: float,
) -> None:
    try:
        profile = json.loads(uploaded_file.getvalue().decode("utf-8"))

        if profile.get("profile_version") != 1:
            raise ValueError("Unsupported or missing profile version.")

        if profile.get("track_name") != track_name:
            raise ValueError(
                f"This profile is for '{profile.get('track_name')}', "
                f"not '{track_name}'."
            )

        saved_length = float(profile.get("track_length_m", 0))

        if abs(saved_length - track_length_m) > 5:
            raise ValueError("Profile track length does not match this data.")

        imported_zones = profile.get("zones")

        if not isinstance(imported_zones, list):
            raise ValueError("Profile zones must be a list.")

        validated_zones = [
            validate_zone(zone, track_length_m)
            for zone in imported_zones
        ]

        zone_names = [
            zone["name"].lower()
            for zone in validated_zones
        ]

        if len(zone_names) != len(set(zone_names)):
            raise ValueError("Profile contains duplicate zone names.")

        st.session_state.zones = validated_zones
        st.success(f"Loaded {len(validated_zones)} zone(s).")

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        st.error(f"Could not load profile: {error}")


def get_lap_data(
    telemetry: pd.DataFrame,
    lap_id: str,
) -> pd.DataFrame:
    return (
        telemetry.loc[
            telemetry["lap_id"].astype(str).eq(str(lap_id))
        ]
        .sort_values("lap_distance")
        .copy()
    )


def filter_to_zone(
    telemetry: pd.DataFrame,
    lap_id: str,
    start_m: float,
    end_m: float,
) -> pd.DataFrame:
    lap_distance_values = pd.to_numeric(
        telemetry["lap_distance"],
        errors="coerce",
    )

    return (
        telemetry.loc[
            telemetry["lap_id"].astype(str).eq(str(lap_id))
            & lap_distance_values.between(
                float(start_m),
                float(end_m),
                inclusive="both",
            )
        ]
        .sort_values("lap_distance")
        .copy()
    )


def zone_metrics(zone_data: pd.DataFrame) -> dict:
    ordered_data = zone_data.sort_values("lap_distance").copy()

    if ordered_data.empty:
        return {
            "zone_time_s": np.nan,
            "entry_speed": np.nan,
            "minimum_speed": np.nan,
            "exit_speed": np.nan,
            "mean_speed": np.nan,
            "mean_throttle": np.nan,
            "mean_front_brake": np.nan,
            "max_front_brake": np.nan,
            "brake_distance_m": np.nan,
            "throttle_distance_m": np.nan,
            "mean_rear_slip": np.nan,
            "rear_slip_event_rate": np.nan,
        }

    sample_count = len(ordered_data)
    window_size = max(1, int(sample_count * 0.10))

    entry_data = ordered_data.head(window_size)
    exit_data = ordered_data.tail(window_size)

    braking_rows = ordered_data.loc[
        ordered_data["brake_0"] >= 0.05
    ]

    throttle_rows = ordered_data.loc[
        ordered_data["throttle"] >= 0.10
    ]

    zone_time_s = (
        float(ordered_data["lap_time"].max())
        - float(ordered_data["lap_time"].min())
    )

    return {
        "zone_time_s": zone_time_s,
        "entry_speed": float(entry_data["speed_magnitude"].mean()),
        "minimum_speed": float(ordered_data["speed_magnitude"].min()),
        "exit_speed": float(exit_data["speed_magnitude"].mean()),
        "mean_speed": float(ordered_data["speed_magnitude"].mean()),
        "mean_throttle": float(ordered_data["throttle"].mean()),
        "mean_front_brake": float(ordered_data["brake_0"].mean()),
        "max_front_brake": float(ordered_data["brake_0"].max()),
        "brake_distance_m": (
            float(braking_rows["lap_distance"].min())
            if not braking_rows.empty
            else np.nan
        ),
        "throttle_distance_m": (
            float(throttle_rows["lap_distance"].min())
            if not throttle_rows.empty
            else np.nan
        ),
        "mean_rear_slip": float(
            ordered_data["wheel_slip_ratio_1"].mean()
        ),
        "rear_slip_event_rate": float(
            ordered_data["rear_slip_event"].mean()
        ),
    }


def build_track_figure(
    lap_data: pd.DataFrame,
    lap_label: str,
    zones: list[dict],
) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=lap_data["world_position_X"],
            y=lap_data["world_position_Y"],
            mode="lines",
            name=lap_label,
            line={"color": UNASSIGNED_COLOUR, "width": 4},
            customdata=lap_data[["lap_distance"]],
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Lap distance: %{customdata[0]:.1f} m"
                "<extra></extra>"
            ),
        )
    )

    for zone in zones:
        zone_data = lap_data.loc[
            lap_data["lap_distance"].between(
                zone["start_m"],
                zone["end_m"],
            )
        ]

        if zone_data.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=zone_data["world_position_X"],
                y=zone_data["world_position_Y"],
                mode="lines",
                name=zone["name"],
                line={"color": zone["colour"], "width": 8},
                customdata=zone_data[["lap_distance"]],
                hovertemplate=(
                    f"<b>{zone['name']}</b><br>"
                    "Lap distance: %{customdata[0]:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_yaxes(scaleanchor="x", scaleratio=1)

    figure.update_layout(
        title=f"Track trajectory — {lap_label}",
        height=650,
        hovermode="closest",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "y": -0.12},
    )

    return figure


def build_comparison_figure(
    reference_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    reference_label: str,
    comparison_label: str,
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
        ("speed_magnitude", "Speed magnitude", 1, "#6B7280"),
        ("throttle", "Throttle", 2, "#2E86DE"),
        ("brake_0", "Front brake", 2, "#E74C3C"),
        ("wheel_slip_ratio_1", "Rear slip ratio", 3, "#8E44AD"),
    ]

    for column, signal_name, row, colour in signals:
        figure.add_trace(
            go.Scatter(
                x=reference_data["lap_distance"],
                y=reference_data[column],
                mode="lines",
                name=f"Reference: {signal_name}",
                line={"color": colour, "width": 2, "dash": "dash"},
            ),
            row=row,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=comparison_data["lap_distance"],
                y=comparison_data[column],
                mode="lines",
                name=f"Comparison: {signal_name}",
                line={"color": colour, "width": 3},
            ),
            row=row,
            col=1,
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

    figure.update_layout(
        title=(
            f"{zone_name}: {comparison_label} vs {reference_label}"
        ),
        height=850,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 85, "b": 80},
        legend={"orientation": "h", "y": -0.12},
    )

    return figure


def build_suggestions(
    telemetry: pd.DataFrame,
    reference_lap_index: int,
    comparison_lap_index: int,
    zones: list[dict],
) -> pd.DataFrame:
    results = []

    for zone in zones:
        reference_data = filter_to_zone(
            telemetry,
            reference_lap_index,
            zone["start_m"],
            zone["end_m"],
        )

        comparison_data = filter_to_zone(
            telemetry,
            comparison_lap_index,
            zone["start_m"],
            zone["end_m"],
        )

        if reference_data.empty or comparison_data.empty:
            continue

        reference = zone_metrics(reference_data)
        comparison = zone_metrics(comparison_data)

        time_delta = (
            comparison["zone_time_s"] - reference["zone_time_s"]
        )
        entry_speed_delta = (
            comparison["entry_speed"] - reference["entry_speed"]
        )
        minimum_speed_delta = (
            comparison["minimum_speed"] - reference["minimum_speed"]
        )
        exit_speed_delta = (
            comparison["exit_speed"] - reference["exit_speed"]
        )
        throttle_delta = (
            comparison["mean_throttle"]
            - reference["mean_throttle"]
        )
        brake_delta = (
            comparison["mean_front_brake"]
            - reference["mean_front_brake"]
        )
        slip_event_delta = (
            comparison["rear_slip_event_rate"]
            - reference["rear_slip_event_rate"]
        )

        brake_point_delta_m = np.nan
        if (
            pd.notna(reference["brake_distance_m"])
            and pd.notna(comparison["brake_distance_m"])
        ):
            brake_point_delta_m = (
                comparison["brake_distance_m"]
                - reference["brake_distance_m"]
            )

        throttle_point_delta_m = np.nan
        if (
            pd.notna(reference["throttle_distance_m"])
            and pd.notna(comparison["throttle_distance_m"])
        ):
            throttle_point_delta_m = (
                comparison["throttle_distance_m"]
                - reference["throttle_distance_m"]
            )

        # Positive score means a greater telemetry difference.
        # Time loss is weighted most heavily.
        impact_score = (
            abs(time_delta) * 12
            + abs(entry_speed_delta) * 0.20
            + abs(minimum_speed_delta) * 0.50
            + abs(exit_speed_delta) * 0.40
            + abs(throttle_delta) * 3
            + abs(brake_delta) * 3
            + abs(slip_event_delta) * 5
        )

        if time_delta <= -0.03:
            status = "Gain"
        elif time_delta >= 0.03:
            status = "Loss"
        else:
            status = "Neutral"

        if time_delta >= 0.05 and exit_speed_delta <= -0.50:
            focus = "Exit-speed loss"
            observation = (
                f"Zone time is {time_delta:+.3f} s versus the reference. "
                f"Exit speed is {exit_speed_delta:+.2f} lower."
            )
            next_check = (
                "Compare throttle opening, rear-slip ratio, and speed "
                "from the apex through the zone exit."
            )

        elif time_delta >= 0.05 and minimum_speed_delta <= -0.50:
            focus = "Minimum-speed loss"
            observation = (
                f"Zone time is {time_delta:+.3f} s slower. "
                f"Minimum speed is {minimum_speed_delta:+.2f} lower."
            )
            next_check = (
                "Compare brake release and the speed trace around the "
                "lowest-speed point."
            )

        elif (
            time_delta >= 0.05
            and pd.notna(brake_point_delta_m)
            and brake_point_delta_m < -10
        ):
            focus = "Earlier braking"
            observation = (
                f"Zone time is {time_delta:+.3f} s slower. "
                f"Braking begins {abs(brake_point_delta_m):.1f} m earlier."
            )
            next_check = (
                "Compare brake-onset position, peak brake pressure, and "
                "minimum speed."
            )

        elif (
            time_delta >= 0.05
            and pd.notna(throttle_point_delta_m)
            and throttle_point_delta_m > 10
        ):
            focus = "Later throttle application"
            observation = (
                f"Zone time is {time_delta:+.3f} s slower. "
                f"Throttle begins {throttle_point_delta_m:.1f} m later."
            )
            next_check = (
                "Compare the first throttle opening with rear-slip ratio "
                "and exit-speed build-up."
            )

        elif slip_event_delta >= 0.05:
            focus = "Traction inconsistency"
            observation = (
                f"Rear-slip event rate is "
                f"{slip_event_delta * 100:+.1f} percentage points higher."
            )
            next_check = (
                "Compare throttle rise and rear wheel-slip traces to find "
                "where traction becomes less consistent."
            )

        elif time_delta <= -0.05:
            focus = "Reference-beating zone"
            observation = (
                f"Zone time is {time_delta:+.3f} s faster. "
                f"Entry / minimum / exit speed deltas are "
                f"{entry_speed_delta:+.2f} / "
                f"{minimum_speed_delta:+.2f} / "
                f"{exit_speed_delta:+.2f}."
            )
            next_check = (
                "Use this zone as a positive reference and identify which "
                "speed or control pattern should be retained."
            )

        else:
            focus = "Small difference"
            observation = (
                f"Zone time is {time_delta:+.3f} s; entry / minimum / "
                f"exit speed deltas are {entry_speed_delta:+.2f} / "
                f"{minimum_speed_delta:+.2f} / "
                f"{exit_speed_delta:+.2f}."
            )
            next_check = (
                "This zone is closely matched; focus analysis effort on "
                "higher-impact zones first."
            )

        results.append(
            {
                "impact_score": impact_score,
                "zone": zone["name"],
                "status": status,
                "focus": focus,
                "zone_time_delta_s": time_delta,
                "entry_speed_delta": entry_speed_delta,
                "minimum_speed_delta": minimum_speed_delta,
                "exit_speed_delta": exit_speed_delta,
                "throttle_delta": throttle_delta,
                "front_brake_delta": brake_delta,
                "slip_event_rate_delta": slip_event_delta,
                "brake_point_delta_m": brake_point_delta_m,
                "throttle_point_delta_m": throttle_point_delta_m,
                "observation": observation,
                "next_check": next_check,
            }
        )

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values(
            ["impact_score", "zone_time_delta_s"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def build_comparison_export(
    zone: dict,
    reference_label: str,
    comparison_label: str,
    reference_lap_time: float,
    comparison_lap_time: float,
    reference: dict,
    comparison: dict,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "zone_name": zone["name"],
                "zone_start_m": zone["start_m"],
                "zone_end_m": zone["end_m"],
                "reference_lap": reference_label,
                "comparison_lap": comparison_label,
                "reference_lap_time_s": reference_lap_time,
                "comparison_lap_time_s": comparison_lap_time,
                "lap_time_delta_s": (
                    comparison_lap_time - reference_lap_time
                ),
                "reference_mean_speed": reference["mean_speed"],
                "comparison_mean_speed": comparison["mean_speed"],
                "mean_speed_delta": (
                    comparison["mean_speed"] - reference["mean_speed"]
                ),
                "reference_mean_throttle": reference["mean_throttle"],
                "comparison_mean_throttle": comparison["mean_throttle"],
                "mean_throttle_delta": (
                    comparison["mean_throttle"]
                    - reference["mean_throttle"]
                ),
                "reference_mean_front_brake": (
                    reference["mean_front_brake"]
                ),
                "comparison_mean_front_brake": (
                    comparison["mean_front_brake"]
                ),
                "mean_front_brake_delta": (
                    comparison["mean_front_brake"]
                    - reference["mean_front_brake"]
                ),
                "reference_mean_rear_slip": (
                    reference["mean_rear_slip"]
                ),
                "comparison_mean_rear_slip": (
                    comparison["mean_rear_slip"]
                ),
                "mean_rear_slip_delta": (
                    comparison["mean_rear_slip"]
                    - reference["mean_rear_slip"]
                ),
                "reference_slip_event_rate": (
                    reference["rear_slip_event_rate"]
                ),
                "comparison_slip_event_rate": (
                    comparison["rear_slip_event_rate"]
                ),
                "slip_event_rate_delta": (
                    comparison["rear_slip_event_rate"]
                    - reference["rear_slip_event_rate"]
                ),
            }
        ]
    ).round(6)


def main() -> None:
    initialise_state()

    try:
        laps, telemetry, data_mode = load_data()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    if laps.empty:
        st.error("No usable laps are available.")
        st.stop()

    laps = create_lap_labels(laps)
    if "lap_index" not in laps.columns and "lapIndex" in laps.columns:
        laps = laps.rename(columns={"lapIndex": "lap_index"})

    if "lap_index" not in laps.columns:
        st.error(
            "The lap summary needs a 'lap_index' column to match telemetry."
        )
        st.stop()

    laps["lap_index"] = pd.to_numeric(
        laps["lap_index"],
        errors="coerce",
    )

    if laps["lap_index"].isna().any():
        st.error("One or more laps have an invalid lap_index value.")
        st.stop()
    # with st.expander("Lap ordering diagnostic", expanded=False):
    #     diagnostic_columns = [
    #         column
    #         for column in [
    #             "display_lap_number",
    #             "lap_index",
    #             "lapIndex",
    #             "lap_number",
    #             "lapnumber",
    #             "lapNum",
    #             "lap_id",
    #             "max_lap_time_s",
    #             "lap_label",
    #         ]
    #         if column in laps.columns
    #     ]

    # st.dataframe(
    #     laps[diagnostic_columns],
    #     use_container_width=True,
    #     hide_index=True,
    # )
    lap_label_to_index = dict(
        zip(laps["lap_label"], laps["lap_index"])
    )
    lap_label_to_id = dict(
        zip(laps["lap_label"], laps["lap_id"])
    )
    lap_id_to_label = dict(
        zip(laps["lap_id"], laps["lap_label"])
    )

    track_length_m = get_track_length(telemetry)

    if "track_id" in telemetry.columns:
        track_name = str(telemetry["track_id"].iloc[0])
    else:
        track_name = DEFAULT_TRACK_NAME

    best_lap = laps.loc[laps["max_lap_time_s"].idxmin()]
    best_lap_label = best_lap["lap_label"]
    best_lap_time = float(best_lap["max_lap_time_s"])

    st.title("🏍️ TracksideTelemetry")
    st.caption(
        "MotoGP 18 telemetry analysis and simulator workflow prototype"
    )

    st.warning(
        "Simulator-data project only. This dashboard does not control "
        "a motorcycle ECU and does not provide real-world riding advice."
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric("Track", track_name)
    metric_2.metric("Usable laps", len(laps))
    metric_3.metric("Best lap", f"{best_lap_time:.3f} s")
    metric_4.metric("Track length", f"{track_length_m:.1f} m")

    st.info(data_mode)

    dashboard_tab, zones_tab, analysis_tab = st.tabs(
        [
            "Dashboard",
            "Zone manager",
            "Lap comparison",
        ]
    )

    with dashboard_tab:
        st.subheader("Lap leaderboard")

        leaderboard = laps.loc[
            :,
            [
                "display_lap_number",
                "max_lap_time_s",
                "distance_coverage_pct",
                "sample_count",
                "max_rear_slip",
            ],
        ].copy()

        leaderboard["Lap"] = leaderboard["display_lap_number"].apply(
            lambda value: f"Lap {int(value)}"
        )

        leaderboard["Gap to best (s)"] = (
            leaderboard["max_lap_time_s"] - best_lap_time
        )

        leaderboard = leaderboard.rename(
            columns={
                "max_lap_time_s": "Lap time (s)",
                "distance_coverage_pct": "Coverage (%)",
                "sample_count": "Samples",
                "max_rear_slip": "Maximum rear-slip ratio",
            }
        )[
            [
                "Lap",
                "Lap time (s)",
                "Gap to best (s)",
                "Coverage (%)",
                "Samples",
                "Maximum rear-slip ratio",
            ]
        ].round(3)

        st.dataframe(
            leaderboard,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Individual lap viewer")

        viewer_label = st.selectbox(
            "Choose a lap to display",
            options=laps["lap_label"].tolist(),
            index=0,
            key="lap_viewer",
        )

        viewer_lap_id = lap_label_to_id[viewer_label]
        viewer_lap_data = get_lap_data(telemetry, viewer_lap_id)

        track_figure = build_track_figure(
            viewer_lap_data,
            viewer_label,
            st.session_state.zones,
        )

        st.plotly_chart(
            track_figure,
            use_container_width=True,
        )

        st.caption(
            "Choose any cleanly labelled lap above. Coloured overlays "
            "appear after zones have been loaded or created."
        )

    with zones_tab:
        st.subheader("Zone manager")

        st.caption(
            "Zones are editable lap-distance sections used for visual "
            "comparison and telemetry suggestions."
        )

        starter_column, clear_column = st.columns(2)

        with starter_column:
            if st.button(
                "Load starter zones",
                use_container_width=True,
            ):
                st.session_state.zones = build_starter_zones(
                    track_length_m
                )
                st.rerun()

        with clear_column:
            if st.button(
                "Clear all zones",
                use_container_width=True,
            ):
                st.session_state.zones = []
                st.rerun()

        with st.form("add_zone_form", clear_on_submit=True):
            name_column, start_column, end_column, colour_column = st.columns(4)

            with name_column:
                name = st.text_input(
                    "Zone name",
                    placeholder="e.g. Turn 1 braking",
                )

            with start_column:
                start_m = st.number_input(
                    "Start distance (m)",
                    min_value=0.0,
                    max_value=float(track_length_m),
                    value=0.0,
                    step=10.0,
                )

            with end_column:
                end_m = st.number_input(
                    "End distance (m)",
                    min_value=0.0,
                    max_value=float(track_length_m),
                    value=min(100.0, float(track_length_m)),
                    step=10.0,
                )

            with colour_column:
                colour = st.color_picker(
                    "Zone colour",
                    value=DEFAULT_ZONE_COLOUR,
                )

            submitted = st.form_submit_button("Add custom zone")

        if submitted:
            add_zone(
                name,
                start_m,
                end_m,
                colour,
                track_length_m,
            )

        st.subheader("Active zones")

        if st.session_state.zones:
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
        else:
            st.info(
                "No zones are active. Load starter zones or create a "
                "custom analysis zone."
            )

        st.subheader("Zone profiles")

        profile_column, import_column = st.columns(2)

        with profile_column:
            profile_json = json.dumps(
                build_zone_profile(
                    track_name,
                    track_length_m,
                ),
                indent=2,
            )

            st.download_button(
                "Download zone profile JSON",
                data=profile_json,
                file_name="telemetry_zone_profile.json",
                mime="application/json",
                disabled=not st.session_state.zones,
                use_container_width=True,
            )

        with import_column:
            uploaded_profile = st.file_uploader(
                "Upload zone profile JSON",
                type=["json"],
            )

            if uploaded_profile is not None:
                signature = (
                    f"{uploaded_profile.name}_{uploaded_profile.size}"
                )

                if (
                    st.session_state.loaded_profile_signature
                    != signature
                ):
                    load_zone_profile(
                        uploaded_profile,
                        track_name,
                        track_length_m,
                    )
                    st.session_state.loaded_profile_signature = signature
                    st.rerun()

    with analysis_tab:
        st.subheader("Choose individual laps")

        st.caption(
            "Select any two separate laps. Neither selection is forced "
            "to be the best lap."
        )

        reference_column, comparison_column = st.columns(2)

        with reference_column:
            reference_label = st.selectbox(
                "Reference lap",
                options=laps["lap_label"].tolist(),
                index=0,
                key="reference_lap",
            )

        comparison_options = [
            label
            for label in laps["lap_label"].tolist()
            if label != reference_label
        ]

        with comparison_column:
            comparison_label = st.selectbox(
                "Comparison lap",
                options=comparison_options,
                index=0,
                key="comparison_lap",
            )

        reference_lap_id = lap_label_to_id[reference_label]
        comparison_lap_id = lap_label_to_id[comparison_label]

        reference_lap_time = float(
            laps.loc[
                laps["lap_id"] == reference_lap_id,
                "max_lap_time_s",
            ].iloc[0]
        )

        comparison_lap_time = float(
            laps.loc[
                laps["lap_id"] == comparison_lap_id,
                "max_lap_time_s",
            ].iloc[0]
        )

        lap_delta = comparison_lap_time - reference_lap_time

        st.info(
            f"Comparing **{comparison_label}** against "
            f"**{reference_label}**. "
            f"Lap-time delta: **{lap_delta:+.3f} s**."
        )

        if not st.session_state.zones:
            st.warning(
                "Load starter zones or add a custom zone in Zone manager "
                "to enable zone analysis."
            )
            st.stop()

        suggestions = build_suggestions(
            telemetry,
            reference_lap_id,
            comparison_lap_id,
            st.session_state.zones,
        )

        st.subheader("Zone comparison findings")

        st.caption(
            "Zones are ranked by telemetry impact. A positive zone-time delta "
            "means the comparison lap spent longer in that zone."
        )

        if suggestions.empty:
            st.info("No comparable telemetry data is available for these zones.")

        else:
            total_zone_loss = suggestions.loc[
                suggestions["zone_time_delta_s"] > 0,
                "zone_time_delta_s",
            ].sum()

            total_zone_gain = suggestions.loc[
                suggestions["zone_time_delta_s"] < 0,
                "zone_time_delta_s",
            ].sum()

            findings_metric_1, findings_metric_2, findings_metric_3 = st.columns(3)

            findings_metric_1.metric(
                "Analysed zone losses",
                f"{total_zone_loss:+.3f} s",
                help="Sum of positive comparison-versus-reference zone-time deltas.",
            )

            findings_metric_2.metric(
                "Analysed zone gains",
                f"{total_zone_gain:+.3f} s",
                help="Sum of negative comparison-versus-reference zone-time deltas.",
            )

            findings_metric_3.metric(
                "Highest-impact zone",
                suggestions.iloc[0]["zone"],
                help="Ranked using time, speed, control-input, and traction deltas.",
            )

            for index, suggestion in suggestions.head(3).iterrows():
                delta = suggestion["zone_time_delta_s"]

                if delta > 0.03:
                    delta_text = f"{delta:+.3f} s slower"
                elif delta < -0.03:
                    delta_text = f"{abs(delta):.3f} s faster"
                else:
                    delta_text = f"{delta:+.3f} s matched"

                with st.expander(
                    f"Priority {index + 1}: {suggestion['zone']} — "
                    f"{suggestion['focus']} ({delta_text})",
                    expanded=index == 0,
                ):
                    comparison_metrics = st.columns(4)

                    comparison_metrics[0].metric(
                        "Zone time delta",
                        f"{suggestion['zone_time_delta_s']:+.3f} s",
                    )

                    comparison_metrics[1].metric(
                        "Entry-speed delta",
                        f"{suggestion['entry_speed_delta']:+.2f}",
                    )

                    comparison_metrics[2].metric(
                        "Minimum-speed delta",
                        f"{suggestion['minimum_speed_delta']:+.2f}",
                    )

                    comparison_metrics[3].metric(
                        "Exit-speed delta",
                        f"{suggestion['exit_speed_delta']:+.2f}",
                    )

                    st.write(f"**Finding:** {suggestion['observation']}")
                    st.write(f"**Trace review:** {suggestion['next_check']}")

            findings_table = suggestions.rename(
                columns={
                    "zone": "Zone",
                    "status": "Status",
                    "focus": "Primary finding",
                    "impact_score": "Impact score",
                    "zone_time_delta_s": "Zone time delta (s)",
                    "entry_speed_delta": "Entry speed delta",
                    "minimum_speed_delta": "Minimum speed delta",
                    "exit_speed_delta": "Exit speed delta",
                    "throttle_delta": "Throttle delta",
                    "front_brake_delta": "Front brake delta",
                    "slip_event_rate_delta": "Slip-event delta",
                    "brake_point_delta_m": "Brake-point delta (m)",
                    "throttle_point_delta_m": "Throttle-point delta (m)",
                }
            )

            st.dataframe(
                findings_table[
                    [
                        "Zone",
                        "Status",
                        "Primary finding",
                        "Zone time delta (s)",
                        "Entry speed delta",
                        "Minimum speed delta",
                        "Exit speed delta",
                        "Throttle delta",
                        "Front brake delta",
                        "Slip-event delta",
                        "Brake-point delta (m)",
                        "Throttle-point delta (m)",
                        "Impact score",
                    ]
                ].round(
                    {
                        "Zone time delta (s)": 3,
                        "Entry speed delta": 2,
                        "Minimum speed delta": 2,
                        "Exit speed delta": 2,
                        "Throttle delta": 3,
                        "Front brake delta": 3,
                        "Slip-event delta": 3,
                        "Brake-point delta (m)": 1,
                        "Throttle-point delta (m)": 1,
                        "Impact score": 3,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        zone_time_chart = (
            suggestions.loc[
                :,
                ["zone", "zone_time_delta_s"],
            ]
            .sort_values("zone_time_delta_s", ascending=True)
        )

        zone_time_figure = go.Figure()

    zone_time_figure.add_trace(
        go.Bar(
            x=zone_time_chart["zone_time_delta_s"],
            y=zone_time_chart["zone"],
            orientation="h",
            marker_color=[
                "#E74C3C" if value > 0 else "#27AE60"
                for value in zone_time_chart["zone_time_delta_s"]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Comparison vs reference: %{x:+.3f} s"
                "<extra></extra>"
            ),
        )
    )

    zone_time_figure.add_vline(
        x=0,
        line_width=1,
        line_color="#6B7280",
    )

    zone_time_figure.update_layout(
        title="Zone-time delta — comparison versus reference",
        xaxis_title="Time delta (s); positive = slower comparison lap",
        yaxis_title="",
        height=max(350, len(zone_time_chart) * 55),
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
    )

    st.plotly_chart(
        zone_time_figure,
        use_container_width=True,
    )
    st.subheader("Selected zone comparison")

    zone_name = st.selectbox(
        "Choose an analysis zone",
        options=[
            zone["name"]
            for zone in st.session_state.zones
        ],
        key="analysis_zone",
    )

    selected_zone = next(
        zone
        for zone in st.session_state.zones
        if zone["name"] == zone_name
    )

    reference_zone_data = filter_to_zone(
        telemetry,
        reference_lap_id,
        selected_zone["start_m"],
        selected_zone["end_m"],
    )

    comparison_zone_data = filter_to_zone(
        telemetry,
        comparison_lap_id,
        selected_zone["start_m"],
        selected_zone["end_m"],
    )

    if reference_zone_data.empty or comparison_zone_data.empty:
        st.error(
            "No telemetry samples were found for this zone and selected "
            "lap pair."
        )

        diagnostic = pd.DataFrame(
            {
                "Lap": ["Reference", "Comparison"],
                "Lap index": [
                    reference_lap_index,
                    comparison_lap_index,
                ],
                "Zone start (m)": [
                    selected_zone["start_m"],
                    selected_zone["start_m"],
                ],
                "Zone end (m)": [
                    selected_zone["end_m"],
                    selected_zone["end_m"],
                ],
                "Samples found": [
                    len(reference_zone_data),
                    len(comparison_zone_data),
                ],
            }
        )

        st.dataframe(
            diagnostic,
            use_container_width=True,
            hide_index=True,
        )

    else:
        reference_metrics = zone_metrics(reference_zone_data)
        comparison_metrics = zone_metrics(comparison_zone_data)

        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Mean speed magnitude",
            f"{comparison_metrics['mean_speed']:.2f}",
            (
                f"{comparison_metrics['mean_speed'] - reference_metrics['mean_speed']:+.2f}"
            ),
        )

        metric_columns[1].metric(
            "Mean throttle",
            f"{comparison_metrics['mean_throttle']:.3f}",
            (
                f"{comparison_metrics['mean_throttle'] - reference_metrics['mean_throttle']:+.3f}"
            ),
        )

        metric_columns[2].metric(
            "Mean front brake",
            f"{comparison_metrics['mean_front_brake']:.3f}",
            (
                f"{comparison_metrics['mean_front_brake'] - reference_metrics['mean_front_brake']:+.3f}"
            ),
        )

        metric_columns[3].metric(
            "Mean rear-slip ratio",
            f"{comparison_metrics['mean_rear_slip']:.3f}",
            (
                f"{comparison_metrics['mean_rear_slip'] - reference_metrics['mean_rear_slip']:+.3f}"
            ),
        )

        metric_columns[4].metric(
            "Rear-slip event rate",
            f"{comparison_metrics['rear_slip_event_rate'] * 100:.1f}%",
            (
                f"{(comparison_metrics['rear_slip_event_rate'] - reference_metrics['rear_slip_event_rate']) * 100:+.1f}%"
            ),
        )

        comparison_figure = build_comparison_figure(
            reference_zone_data,
            comparison_zone_data,
            reference_label,
            comparison_label,
            selected_zone["name"],
        )

        st.plotly_chart(
            comparison_figure,
            use_container_width=True,
        )

    export_data = build_comparison_export(
        selected_zone,
        reference_label,
        comparison_label,
        reference_lap_time,
        comparison_lap_time,
        reference_metrics,
        comparison_metrics,
    )

    safe_zone_name = (
        selected_zone["name"]
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    st.download_button(
        "Download selected-zone comparison CSV",
        data=export_data.to_csv(index=False).encode("utf-8"),
        file_name=f"lap_comparison_{safe_zone_name}.csv",
        mime="text/csv",
    )

    export_data = build_comparison_export(
        selected_zone,
        reference_label,
        comparison_label,
        reference_lap_time,
        comparison_lap_time,
        reference_metrics,
        comparison_metrics,
    )

    safe_zone_name = (
        selected_zone["name"]
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    st.download_button(
        "Download selected-zone comparison CSV",
        data=export_data.to_csv(index=False).encode("utf-8"),
        file_name=f"lap_comparison_{safe_zone_name}.csv",
        mime="text/csv",
        key=(
            f"download_zone_comparison_"
            f"{reference_lap_id}_"
            f"{comparison_lap_id}_"
            f"{safe_zone_name}"
        ),
    )


if __name__ == "__main__":
    main()