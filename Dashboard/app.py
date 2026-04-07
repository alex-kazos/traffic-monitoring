from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html

from data_pipeline import ALERT_THRESHOLD_KMH, BIN_SECONDS, build_alerts, load_vehicle_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
	"segment_id",
	"total_frames",
	"entry_frame",
	"speed_kmh",
	"carriageway",
	"vehicle_type",
	"vehicle_id",
	"segment_sort_key",
	"frames_before_segment",
	"absolute_frame",
	"timestamp_seconds",
	"timestamp_minutes",
	"bin_idx",
	"time_bin",
]


def _empty_vehicle_frame() -> pd.DataFrame:
	return pd.DataFrame(columns=EXPECTED_COLUMNS)


def _load_dashboard_data() -> tuple[pd.DataFrame, str | None]:
	try:
		return load_vehicle_data(PROJECT_ROOT), None
	except Exception as exc:
		return _empty_vehicle_frame(), f"SQL data could not be loaded: {exc}"


RAW_DF, DATA_STATUS_MESSAGE = _load_dashboard_data()

CARRIAGEWAYS = sorted(RAW_DF["carriageway"].dropna().astype(str).unique())
VEHICLE_TYPES = sorted(RAW_DF["vehicle_type"].dropna().astype(str).unique())
BIN_MINUTES = BIN_SECONDS // 60
MIN_BIN_START_MINUTE = 0
MAX_BIN_START_MINUTE = int((RAW_DF["timestamp_seconds"].max() // BIN_SECONDS) * BIN_MINUTES) if not RAW_DF.empty else BIN_MINUTES
MAX_BIN_END_MINUTE = max(MIN_BIN_START_MINUTE + BIN_MINUTES, MAX_BIN_START_MINUTE + BIN_MINUTES)


def _bin_label(bin_start_minute: int) -> str:
	return f"{bin_start_minute}-{bin_start_minute + BIN_MINUTES}min"


def _slider_marks(min_minute: int, max_minute: int) -> dict[int, str]:
	marks = {minute: _bin_label(minute) for minute in range(min_minute, max_minute, BIN_MINUTES)}
	marks[max_minute] = f"{max_minute}m"
	return marks


app = Dash(__name__)
app.title = "Traffic Speed Dashboard"

app.layout = html.Div(
	[
		html.H2("Traffic Speed Dashboard"),
		html.P("Move the slider to inspect average speed and over-speed alerts across time."),
		html.Div(
			DATA_STATUS_MESSAGE,
			style={"marginBottom": "12px", "padding": "10px 12px", "borderRadius": "6px", "backgroundColor": "#fff4ce", "color": "#5c3b00"},
		) if DATA_STATUS_MESSAGE else html.Div(),
		html.Div(
			[
				html.Label("Carriageway"),
				dcc.Dropdown(
					id="carriageway-filter",
					options=[{"label": item, "value": item} for item in CARRIAGEWAYS],
					value=CARRIAGEWAYS,
					multi=True,
					clearable=False,
				),
			],
			style={"maxWidth": "500px", "marginBottom": "12px"},
		),
		html.Div(
			[
				html.Label("Vehicle type"),
				dcc.Dropdown(
					id="vehicle-type-filter",
					options=[{"label": item, "value": item} for item in VEHICLE_TYPES],
					value=VEHICLE_TYPES,
					multi=True,
					clearable=False,
				),
			],
			style={"maxWidth": "500px", "marginBottom": "14px"},
		),
		html.Label("Time window (5-minute bins)"),
		dcc.RangeSlider(
			id="time-range-slider",
			min=MIN_BIN_START_MINUTE,
			max=MAX_BIN_END_MINUTE,
			step=BIN_MINUTES,
			value=[MIN_BIN_START_MINUTE, MAX_BIN_END_MINUTE],
			marks=_slider_marks(MIN_BIN_START_MINUTE, MAX_BIN_END_MINUTE),
			allowCross=False,
			tooltip={"always_visible": False, "placement": "bottom"},
		),
		dcc.Graph(id="avg-speed-timeseries", style={"height": "470px"}),
		dcc.Graph(id="avg-speed-current-bin", style={"height": "430px"}),
		html.H4("Alerts (>130 km/h)", style={"marginTop": "10px"}),
		html.Div(id="alerts-summary", style={"marginBottom": "8px"}),
		dash_table.DataTable(
			id="alerts-table",
			columns=[
				{"name": "Time", "id": "timestamp_mmss"},
				{"name": "Vehicle ID", "id": "vehicle_id"},
				{"name": "Carriageway", "id": "carriageway"},
				{"name": "Vehicle Type", "id": "vehicle_type"},
				{"name": "Speed (km/h)", "id": "speed_kmh"},
				{"name": "Segment", "id": "segment_id"},
			],
			page_size=12,
			style_table={"overflowX": "auto"},
			style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "Arial", "fontSize": "13px"},
			style_header={"fontWeight": "bold"},
		),
	],
	style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
)


@app.callback(
	Output("avg-speed-timeseries", "figure"),
	Output("avg-speed-current-bin", "figure"),
	Output("alerts-summary", "children"),
	Output("alerts-table", "data"),
	Input("time-range-slider", "value"),
	Input("carriageway-filter", "value"),
	Input("vehicle-type-filter", "value"),
)
def update_dashboard(selected_range: list[int], selected_carriageways: list[str], selected_vehicle_types: list[str]):
	if not selected_range or len(selected_range) != 2:
		start_minute, end_minute = MIN_BIN_START_MINUTE, MAX_BIN_END_MINUTE
	else:
		start_minute, end_minute = sorted(int(value) for value in selected_range)

	selected_carriageways = cast(list[str], selected_carriageways or CARRIAGEWAYS)
	selected_vehicle_types = cast(list[str], selected_vehicle_types or VEHICLE_TYPES)
	if end_minute <= start_minute:
		end_minute = start_minute + BIN_MINUTES

	filtered_raw = RAW_DF[
		RAW_DF["carriageway"].isin(selected_carriageways)
		& RAW_DF["vehicle_type"].isin(selected_vehicle_types)
		& (RAW_DF["timestamp_minutes"] >= start_minute)
		& (RAW_DF["timestamp_minutes"] < end_minute)
	].copy()
	filtered_raw["bin_start_minute"] = (filtered_raw["timestamp_seconds"] // BIN_SECONDS).astype(int) * BIN_MINUTES
	filtered_raw = filtered_raw[
		(filtered_raw["bin_start_minute"] >= start_minute)
		& (filtered_raw["bin_start_minute"] < end_minute)
	]

	if filtered_raw.empty:
		line_fig = px.line(title="Average speed over time")
		line_fig.add_annotation(text="No data for this filter selection.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
	else:
		in_selected_window = (
			filtered_raw.assign(minute_start=filtered_raw["timestamp_minutes"].floordiv(1).astype(int))
			.groupby(["minute_start", "carriageway", "vehicle_type"], as_index=False)
			.agg(avg_speed_kmh=("speed_kmh", "mean"))
			.sort_values("minute_start")
		)
		line_fig = px.line(
			in_selected_window,
			x="minute_start",
			y="avg_speed_kmh",
			color="carriageway",
			line_dash="vehicle_type",
			markers=True,
			hover_data={"vehicle_type": True, "carriageway": True, "avg_speed_kmh": ":.2f", "minute_start": True},
			title="Average speed by carriageway and vehicle type (selected window)",
			labels={"minute_start": "Time (minutes)", "avg_speed_kmh": "Average Speed (km/h)"},
		)
		line_fig.add_vline(x=start_minute, line_width=2, line_dash="dash", line_color="black")
		line_fig.add_vline(x=end_minute, line_width=2, line_dash="dash", line_color="black")

	line_fig.update_layout(legend_title_text="Carriageway / Vehicle")

	if filtered_raw.empty:
		bar_fig = px.bar(title=f"Stacked average speed from {start_minute}m to {end_minute}m")
		bar_fig.add_annotation(text="No records in selected 5-minute bins.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
	else:
		stacked_avg = (
			filtered_raw.groupby(["bin_start_minute", "carriageway", "vehicle_type"], as_index=False)
			.agg(avg_speed_kmh=("speed_kmh", "mean"))
			.sort_values(["bin_start_minute", "carriageway", "vehicle_type"])
		)
		bin_starts = list(range(start_minute, end_minute, BIN_MINUTES))
		carriageway_order = sorted(stacked_avg["carriageway"].astype(str).unique())
		vehicle_order = sorted(stacked_avg["vehicle_type"].astype(str).unique())
		full_index = pd.MultiIndex.from_product(
			[bin_starts, carriageway_order, vehicle_order],
			names=["bin_start_minute", "carriageway", "vehicle_type"],
		)
		stacked_avg = (
			stacked_avg.set_index(["bin_start_minute", "carriageway", "vehicle_type"])
			.reindex(full_index, fill_value=0)
			.reset_index()
		)
		stacked_avg["time_bin"] = stacked_avg["bin_start_minute"].map(_bin_label)
		stacked_avg["time_bin"] = pd.Categorical(
			stacked_avg["time_bin"],
			categories=[_bin_label(minute) for minute in bin_starts],
			ordered=True,
		)
		bar_fig = px.bar(
			stacked_avg,
			x="time_bin",
			y="avg_speed_kmh",
			color="vehicle_type",
			facet_col="carriageway",
			category_orders={"carriageway": carriageway_order, "vehicle_type": vehicle_order},
			title=f"Stacked average speed by vehicle type and carriageway (5-minute bins {start_minute}-{end_minute} min)",
			labels={"time_bin": "Time Bin (5-minute)", "avg_speed_kmh": "Average Speed (km/h)", "vehicle_type": "Vehicle Type"},
		)
		bar_fig.update_layout(barmode="stack", legend_title_text="Vehicle Type")

	filtered_alerts = build_alerts(filtered_raw, threshold_kmh=ALERT_THRESHOLD_KMH)
	last_bin_start = max(end_minute - BIN_MINUTES, start_minute)
	current_bin_alerts_count = int(
		(
			(filtered_alerts["timestamp_minutes"] >= last_bin_start)
			& (filtered_alerts["timestamp_minutes"] < end_minute)
		).sum()
	) if not filtered_alerts.empty else 0
	summary = (
		f"Threshold: {ALERT_THRESHOLD_KMH:.0f} km/h | "
		f"Alerts in last 5-minute bin: {current_bin_alerts_count} | "
		f"Window: {start_minute}m to {end_minute}m | "
		f"Total alerts in window: {len(filtered_alerts)}"
	)

	table_columns = ["timestamp_mmss", "vehicle_id", "carriageway", "vehicle_type", "speed_kmh", "segment_id"]
	table_data = filtered_alerts[table_columns].head(500).to_dict("records") if not filtered_alerts.empty else []

	return line_fig, bar_fig, summary, table_data


if __name__ == "__main__":
	app.run(debug=True)