"""
Local launcher for *_local traffic monitoring flow.

Examples:
  python run_local.py --distribution-only
  python run_local.py --partition 2
  python run_local.py --glob "Downloads/Segments/*_part_00*.mp4"
"""
import argparse
import logging
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local traffic monitoring debug flow")
    parser.add_argument("--partitions", type=int, default=4, help="Number of simulated partitions")
    parser.add_argument("--partition", type=int, default=-1, help="Run only this partition (default: all)")
    parser.add_argument("--distribution-only", action="store_true", help="Only print file-to-partition mapping")
    parser.add_argument(
        "--glob",
        dest="segments_glob",
        default="Downloads/Segments/*_part_*.mp4",
        help="Glob pattern for local segment files",
    )
    parser.add_argument("--backend", default="sqlite", choices=["sqlite", "mssql"], help="DB backend")
    parser.add_argument(
        "--sqlite-db",
        default="Data/traffic_debug.sqlite",
        help="SQLite database path when --backend=sqlite",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("uamqp").setLevel(logging.WARNING)

    # Set env before importing monitor_traffic_local/config_local.
    os.environ["LOCAL_DEBUG_MODE"] = "1"
    os.environ["DEBUG_PARTITIONS"] = str(args.partitions)
    os.environ["DEBUG_TARGET_PARTITION"] = str(args.partition)
    os.environ["DEBUG_DISTRIBUTION_ONLY"] = "1" if args.distribution_only else "0"
    os.environ["DEBUG_LOCAL_SEGMENTS_GLOB"] = args.segments_glob
    os.environ["DB_BACKEND"] = args.backend
    os.environ["SQLITE_DB_PATH"] = args.sqlite_db

    from monitor_traffic_local import main as local_main

    local_main()


if __name__ == "__main__":
    main()


