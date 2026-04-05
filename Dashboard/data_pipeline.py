from __future__ import annotations

from pathlib import Path
import glob
from typing import Iterable

import pandas as pd

FPS = 25.0
BIN_SECONDS = 300
ALERT_THRESHOLD_KMH = 130.0


def discover_csv_files(project_root: Path) -> list[Path]:
    patterns = [
        project_root / "Downloads" / "Data" / "vehicle_speeds_*.csv",
        project_root / "Downloads" / "DATA" / "vehicle_speeds_*.csv",
    ]

    files: set[Path] = set()
    for pattern in patterns:
        for file_path in glob.glob(str(pattern)):
            files.add(Path(file_path).resolve())

    return sorted(files)


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _segment_sort_key(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(0).astype(int)


def load_vehicle_data(project_root: Path) -> pd.DataFrame:
    files = discover_csv_files(project_root)
    if not files:
        raise FileNotFoundError("No vehicle_speeds_*.csv files found under Downloads/Data or Downloads/DATA.")

    df = pd.concat((pd.read_csv(str(file)) for file in files), ignore_index=True)
    if df.empty:
        raise ValueError("CSV files were found but no rows were loaded.")

    required_columns = {"segment_id", "total_frames", "entry_frame", "speed_kmh", "carriageway", "vehicle_type", "vehicle_id"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = _coerce_numeric(df, ["total_frames", "entry_frame", "speed_kmh", "entry_timestamp_s"])
    df = df.dropna(subset=["total_frames", "entry_frame", "speed_kmh"]).copy()

    df["segment_sort_key"] = _segment_sort_key(df["segment_id"])
    segment_frames = (
        df.sort_values(["segment_sort_key", "segment_id"])
        .groupby("segment_id", as_index=False)
        .first()[["segment_id", "segment_sort_key", "total_frames"]]
        .sort_values(["segment_sort_key", "segment_id"])
    )

    segment_frames["frames_before_segment"] = segment_frames["total_frames"].cumsum().shift(fill_value=0)
    frames_before_lookup = segment_frames.set_index("segment_id")["frames_before_segment"]

    df["frames_before_segment"] = df["segment_id"].map(frames_before_lookup)
    df["absolute_frame"] = df["frames_before_segment"] + df["entry_frame"]
    df["timestamp_seconds"] = df["absolute_frame"] / FPS
    df["timestamp_minutes"] = df["timestamp_seconds"] / 60.0
    df["bin_idx"] = (df["timestamp_seconds"] // BIN_SECONDS).astype(int)
    df["time_bin"] = df["bin_idx"].map(lambda idx: f"{idx * 5}-{(idx + 1) * 5}min")

    return df


def build_avg_speed_by_bin(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["bin_idx", "time_bin", "carriageway", "vehicle_type"], as_index=False)
        .agg(avg_speed_kmh=("speed_kmh", "mean"), observations=("speed_kmh", "size"))
        .sort_values("bin_idx")
    )
    grouped["avg_speed_kmh"] = grouped["avg_speed_kmh"].round(2)
    return grouped


def build_alerts(df: pd.DataFrame, threshold_kmh: float = ALERT_THRESHOLD_KMH) -> pd.DataFrame:
    alerts = df[df["speed_kmh"] > threshold_kmh].copy()
    if alerts.empty:
        return alerts

    alerts = alerts.sort_values("timestamp_seconds", ascending=False)
    alerts["speed_kmh"] = alerts["speed_kmh"].round(2)
    alerts["timestamp_mmss"] = pd.to_timedelta(alerts["timestamp_seconds"], unit="s").astype(str)
    return alerts



