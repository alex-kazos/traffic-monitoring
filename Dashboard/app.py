from __future__ import annotations

import math
from pathlib import Path

import plotly.express as px
from dash import Dash, Input, Output, dash_table, dcc, html

from data_pipeline import ALERT_THRESHOLD_KMH, build_alerts, load_vehicle_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DF = load_vehicle_data(PROJECT_ROOT)

CARRIAGEWAYS = sorted(RAW_DF["carriageway"].dropna().astype(str).unique())
VEHICLE_TYPES = sorted(RAW_DF["vehicle_type"].dropna().astype(str).unique())
MIN_MINUTE = 0
MAX_MINUTE = int(math.ceil(RAW_DF["timestamp_minutes"].max()))


def _minute_label(value: int) -> str:
	return f"{value} min"


def _slider_marks(min_minute: int, max_minute: int) -> dict[int, str]:
	span = max_minute - min_minute
	step = max(1, span // 12)
	marks = {minute: _minute_label(minute) for minute in range(min_minute, max_minute + 1, step)}
	if max_minute not in marks:
		marks[max_minute] = _minute_label(max_minute)
	return marks


app = Dash(__name__)
app.title = "Traffic Speed Dashboard"

app.layout = html.Div(
	[
		html.H2("Traffic Speed Dashboard"),
		html.P("Move the slider to inspect average speed and over-speed alerts across time."),
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
		html.Label("Time window (minutes)"),
		dcc.RangeSlider(
			id="time-range-slider",
			min=MIN_MINUTE,
			max=MAX_MINUTE,
			step=1,
			value=[MIN_MINUTE, MAX_MINUTE],
			marks=_slider_marks(MIN_MINUTE, MAX_MINUTE),
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
		start_minute, end_minute = MIN_MINUTE, MAX_MINUTE
	else:
		start_minute, end_minute = sorted(selected_range)

	selected_carriageways = selected_carriageways or CARRIAGEWAYS
	selected_vehicle_types = selected_vehicle_types or VEHICLE_TYPES

	filtered_raw = RAW_DF[
		RAW_DF["carriageway"].isin(selected_carriageways)
		& RAW_DF["vehicle_type"].isin(selected_vehicle_types)
		& (RAW_DF["timestamp_minutes"] >= start_minute)
		& (RAW_DF["timestamp_minutes"] <= end_minute)
	].copy()

	if filtered_raw.empty:
		line_fig = px.line(title="Average speed over time")
		line_fig.add_annotation(text="No data for this filter selection.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
	else:
		filtered_raw["minute_start"] = filtered_raw["timestamp_minutes"].floordiv(1).astype(int)
		in_selected_window = (
			filtered_raw.groupby(["minute_start", "carriageway", "vehicle_type"], as_index=False)
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
		bar_fig = px.bar(title=f"Average speed from {_minute_label(start_minute)} to {_minute_label(end_minute)}")
		bar_fig.add_annotation(text="No records in selected minute window.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
	else:
		current_bin_avg = (
			filtered_raw.groupby(["carriageway", "vehicle_type"], as_index=False)
			.agg(avg_speed_kmh=("speed_kmh", "mean"))
			.sort_values(["carriageway", "vehicle_type"])
		)
		current_bin_avg = current_bin_avg.copy()
		current_bin_avg["group"] = current_bin_avg["carriageway"] + " - " + current_bin_avg["vehicle_type"]
		bar_fig = px.bar(
			current_bin_avg,
			x="group",
			y="avg_speed_kmh",
			color="carriageway",
			text="avg_speed_kmh",
			title=f"Average speed from {_minute_label(start_minute)} to {_minute_label(end_minute)}",
			labels={"group": "Carriageway / Vehicle Type", "avg_speed_kmh": "Average Speed (km/h)"},
		)
		bar_fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

	filtered_alerts = build_alerts(filtered_raw, threshold_kmh=ALERT_THRESHOLD_KMH)

	current_bin_alerts_count = int(
		(
			(filtered_alerts["timestamp_minutes"] >= end_minute)
			& (filtered_alerts["timestamp_minutes"] < (end_minute + 1))
		).sum()
	) if not filtered_alerts.empty else 0
	summary = (
		f"Threshold: {ALERT_THRESHOLD_KMH:.0f} km/h | "
		f"Alerts in minute {end_minute}: {current_bin_alerts_count} | "
		f"Window: {_minute_label(start_minute)} to {_minute_label(end_minute)} | "
		f"Total alerts in window: {len(filtered_alerts)}"
	)

	table_columns = ["timestamp_mmss", "vehicle_id", "carriageway", "vehicle_type", "speed_kmh", "segment_id"]
	table_data = filtered_alerts[table_columns].head(500).to_dict("records") if not filtered_alerts.empty else []

	return line_fig, bar_fig, summary, table_data


if __name__ == "__main__":
	app.run(debug=True)






