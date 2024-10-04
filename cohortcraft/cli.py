"""Command-line interface.

Examples
--------
    cohortcraft generate --customers 10000
    cohortcraft run --regenerate
    cohortcraft score --model outputs/models/xgboost.joblib --out scores.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cohortcraft.config import Config


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(args, "customers", None):
        cfg.n_customers = args.customers
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "output_dir", None):
        cfg.output_dir = Path(args.output_dir)
        cfg.db_path = cfg.output_dir / "cohortcraft.db"
    if getattr(args, "snapshot", None):
        cfg.snapshot_date = args.snapshot
    return cfg


def cmd_generate(args: argparse.Namespace) -> int:
    from cohortcraft.pipeline import generate_data

    generate_data(_config_from_args(args))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from cohortcraft.pipeline import run
    from cohortcraft.report import write_report

    cfg = _config_from_args(args)
    outputs = run(cfg, regenerate=args.regenerate)
    report = write_report(outputs, cfg.output_dir / "report.md")
    print(outputs.metrics.to_string(index=False))
    print(f"\nBest model: {outputs.best_model}")
    print(f"Segments: {', '.join(outputs.segments['name'])}")
    print(f"Report written to {report}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from cohortcraft.data import db
    from cohortcraft.features import build_features
    from cohortcraft.propensity.train import load_model, score_customers

    cfg = _config_from_args(args)
    model = load_model(args.model)
    conn = db.connect(cfg.db_path)
    try:
        feats = build_features(conn, cfg.snapshot_date)
    finally:
        conn.close()
    scores = score_customers(model, feats)
    scores.to_csv(args.out, index=False)
    print(f"Scored {len(scores):,} customers -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cohortcraft", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int)
        p.add_argument("--output-dir")
        p.add_argument("--snapshot", help="Snapshot date YYYY-MM-DD")

    g = sub.add_parser("generate", help="Generate synthetic data into SQLite")
    g.add_argument("--customers", type=int)
    common(g)
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="Run the full modeling pipeline")
    r.add_argument("--customers", type=int)
    r.add_argument("--regenerate", action="store_true", help="Rebuild the database first")
    common(r)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="Score customers with a saved propensity model")
    s.add_argument("--model", required=True)
    s.add_argument("--out", default="scores.csv")
    common(s)
    s.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
