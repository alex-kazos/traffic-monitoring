from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

FPS = 25.0
BIN_SECONDS = 300
ALERT_THRESHOLD_KMH = 130.0
_REQUIRED_COLUMNS = {
    "segment_id",
    "total_frames",
    "entry_frame",
    "speed_kmh",
    "carriageway",
    "vehicle_type",
    "vehicle_id",
}


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _segment_sort_key(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(0).astype(int)


def _validate_identifier(name: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid SQL {label}: {name!r}")
    return name


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_installed_sql_server_drivers() -> list[str]:
    try:
        import pyodbc  # type: ignore[import-not-found]
    except ImportError:
        return []

    return [driver for driver in pyodbc.drivers() if "sql server" in driver.lower()]


def _extract_driver(connection_string: str) -> str:
    match = re.search(r"DRIVER\s*=\s*\{?([^;{}]+)}?", connection_string, flags=re.IGNORECASE)
    return _normalize_text(match.group(1)) if match else ""


def _replace_driver(connection_string: str, new_driver: str) -> str:
    replacement = f"DRIVER={{{new_driver}}}"
    if re.search(r"DRIVER\s*=", connection_string, flags=re.IGNORECASE):
        return re.sub(r"DRIVER\s*=\s*\{?[^;{}]+}?", replacement, connection_string, flags=re.IGNORECASE)
    return f"{replacement};{connection_string}"


def _strip_connection_attributes(connection_string: str, attributes: list[str]) -> str:
    updated = connection_string
    for attribute in attributes:
        updated = re.sub(rf"{re.escape(attribute)}\s*=\s*[^;]*;?", "", updated, flags=re.IGNORECASE)
    return updated


def _resolve_sql_driver(preferred_driver: str) -> str:
    installed = _get_installed_sql_server_drivers()
    if not installed:
        return preferred_driver

    normalized = {item.lower(): item for item in installed}
    if preferred_driver.lower() in normalized:
        return normalized[preferred_driver.lower()]

    for candidate in ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server", "SQL Server"]:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return installed[0]


def _normalize_connection_string_driver(connection_string: str, preferred_driver: str) -> str:
    resolved_driver = _resolve_sql_driver(preferred_driver)
    current_driver = _extract_driver(connection_string)
    normalized = connection_string
    if not current_driver or current_driver.lower() != resolved_driver.lower():
        normalized = _replace_driver(connection_string, resolved_driver)

    if resolved_driver.lower() == "sql server":
        normalized = _strip_connection_attributes(normalized, ["Encrypt", "TrustServerCertificate", "Connection Timeout"])
    return normalized


def _load_config_local(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "config_local.py"
    if not config_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("dashboard_config_local", str(config_path))
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return {}

    return {
        "connection_string": getattr(module, "connection_string", ""),
        "server": getattr(module, "AZURE_SQL_SERVER_NAME", ""),
        "database": getattr(module, "AZURE_SQL_DATABASE_NAME", ""),
        "username": getattr(module, "AZURE_SQL_USER_NAME", ""),
        "password": getattr(module, "AZURE_SQL_PASSWORD", ""),
    }


def _build_connection_string(project_root: Path) -> str:

    server = _normalize_text(os.getenv("AZURE_SQL_SERVER_NAME", ""))
    database = _normalize_text(os.getenv("AZURE_SQL_DATABASE_NAME", ""))
    username = _normalize_text(os.getenv("AZURE_SQL_USER_NAME", ""))
    password = _normalize_text(os.getenv("AZURE_SQL_PASSWORD", ""))

    if not (server and database and username and password):
        config_local = _load_config_local(project_root)
        direct = _normalize_text(config_local.get("connection_string", ""))
        if direct:
            preferred_driver = _normalize_text(os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server"))
            return _normalize_connection_string_driver(direct, preferred_driver)

        server = server or _normalize_text(config_local.get("server", ""))
        database = database or _normalize_text(config_local.get("database", ""))
        username = username or _normalize_text(config_local.get("username", ""))
        password = password or _normalize_text(config_local.get("password", ""))

    missing = [
        key
        for key, value in {
            "AZURE_SQL_SERVER_NAME": server,
            "AZURE_SQL_DATABASE_NAME": database,
            "AZURE_SQL_USER_NAME": username,
            "AZURE_SQL_PASSWORD": password,
        }.items()
        if not value
    ]

    driver = _resolve_sql_driver(_normalize_text(os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")))
    encrypt = _normalize_text(os.getenv("AZURE_SQL_ENCRYPT", "yes"))
    trust_cert = _normalize_text(os.getenv("AZURE_SQL_TRUST_SERVER_CERTIFICATE", "no"))
    timeout = _normalize_text(os.getenv("AZURE_SQL_CONNECTION_TIMEOUT", "30"))

    connection_string = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_cert};"
        f"Connection Timeout={timeout};"
    )
    if driver.lower() == "sql server":
        connection_string = _strip_connection_attributes(connection_string, ["Encrypt", "TrustServerCertificate", "Connection Timeout"])
    return connection_string


def _load_from_sql_server(project_root: Path) -> pd.DataFrame:
    try:
        import pyodbc  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyodbc is required to load dashboard data from SQL Server.") from exc

    schema = _validate_identifier(os.getenv("TRAFFIC_SQL_SCHEMA", "dbo").strip(), "schema")
    table = _validate_identifier(os.getenv("TRAFFIC_SQL_TABLE", "vehicle_speeds").strip(), "table")

    query = (
        "SELECT "
        "segment_id, total_frames, entry_frame, speed_kmh, carriageway, vehicle_type, vehicle_id "
        f"FROM {schema}.{table};"
    )

    connection_string = _build_connection_string(project_root)
    with pyodbc.connect(connection_string) as conn:
        return pd.read_sql_query(query, conn)


def load_vehicle_data(project_root: Path) -> pd.DataFrame:
    df = _load_from_sql_server(project_root)

    if df.empty:
        raise ValueError("SQL query returned no rows from vehicle speed data.")

    missing = _REQUIRED_COLUMNS.difference(df.columns)
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
