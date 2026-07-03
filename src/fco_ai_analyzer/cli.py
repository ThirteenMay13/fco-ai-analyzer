from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import FCOAnalyzer
from .reporting import ReportExporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCO AI Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse and persist a raw log file")
    ingest.add_argument("log_path", type=Path, help="Path to a raw enhancement log text file")
    ingest.add_argument("--database", type=Path, default=Path("fco_ai_analyzer.sqlite3"), help="SQLite database path")

    analyze = subparsers.add_parser("analyze", help="Analyze a raw log file and print recommendation")
    analyze.add_argument("log_path", type=Path, help="Path to a raw enhancement log text file")
    analyze.add_argument("--pattern", type=str, default="", help="Override the current pattern as comma-separated tokens")
    analyze.add_argument("--database", type=Path, default=Path("fco_ai_analyzer.sqlite3"), help="SQLite database path")

    train = subparsers.add_parser("train", help="Train ML models from a raw log file")
    train.add_argument("log_path", type=Path, help="Path to a raw enhancement log text file")
    train.add_argument("--database", type=Path, default=Path("fco_ai_analyzer.sqlite3"), help="SQLite database path")

    export = subparsers.add_parser("export", help="Export HTML, Excel, and PDF reports")
    export.add_argument("log_path", type=Path, help="Path to a raw enhancement log text file")
    export.add_argument("--output-dir", type=Path, default=Path("reports"), help="Output directory")
    export.add_argument("--database", type=Path, default=Path("fco_ai_analyzer.sqlite3"), help="SQLite database path")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    analyzer = FCOAnalyzer(args.database)
    raw_text = args.log_path.read_text(encoding="utf-8")

    if args.command == "ingest":
        analyzer.ingest_text(raw_text)
        print(f"ingested={args.log_path}")
        return 0

    if args.command == "train":
        bundle = analyzer.train_models_from_text(raw_text)
        print(f"trained_models={list(bundle.models.keys())}")
        return 0

    if args.command == "analyze":
        current_pattern = tuple(int(token) for token in args.pattern.split(",") if token.strip()) if args.pattern else None
        recommendation = analyzer.analyze_text(raw_text, current_pattern=current_pattern)
        print(f"pattern={recommendation.current_pattern}")
        print(f"success_probability={recommendation.success_probability:.2%}")
        print(f"confidence={recommendation.confidence}")
        print(f"recommendation={recommendation.recommendation}")
        for line in recommendation.rationale:
            print(f"- {line}")
        return 0

    if args.command == "export":
        report = analyzer.build_report(raw_text)
        exporter = ReportExporter()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        exporter.export_html(args.output_dir / "report.html", report["summary"], report["patterns"], report["recommendation"])
        exporter.export_excel(args.output_dir / "report.xlsx", report["summary"], report["patterns"], report["recommendation"])
        exporter.export_pdf(args.output_dir / "report.pdf", report["summary"], report["patterns"], report["recommendation"])
        print(f"exported_to={args.output_dir}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
