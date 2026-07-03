from __future__ import annotations

from pathlib import Path

from fco_ai_analyzer.parser import LogParser
from fco_ai_analyzer.pipeline import FCOAnalyzer
from fco_ai_analyzer.reporting import ReportExporter


def seed_database(log_path: Path, database_path: Path) -> None:
    """Parse a log file and persist it to the database."""
    raw_text = log_path.read_text(encoding="utf-8")
    analyzer = FCOAnalyzer(database_path)
    analyzer.ingest_text(raw_text)
    print(f"✓ Ingested {log_path} into {database_path}")


def train_models(log_path: Path, database_path: Path) -> None:
    """Train ML models from a log file."""
    raw_text = log_path.read_text(encoding="utf-8")
    analyzer = FCOAnalyzer(database_path)
    model_bundle = analyzer.train_models_from_text(raw_text)
    print(f"✓ Trained models: {list(model_bundle.models.keys())}")


def generate_reports(log_path: Path, database_path: Path, output_dir: Path) -> None:
    """Generate HTML, Excel, and PDF reports from a log file."""
    raw_text = log_path.read_text(encoding="utf-8")
    analyzer = FCOAnalyzer(database_path)
    report = analyzer.build_report(raw_text)
    exporter = ReportExporter()
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter.export_html(output_dir / "report.html", report["summary"], report["patterns"], report["recommendation"])
    print(f"✓ HTML report: {output_dir / 'report.html'}")

    exporter.export_excel(output_dir / "report.xlsx", report["summary"], report["patterns"], report["recommendation"])
    print(f"✓ Excel report: {output_dir / 'report.xlsx'}")

    exporter.export_pdf(output_dir / "report.pdf", report["summary"], report["patterns"], report["recommendation"])
    print(f"✓ PDF report: {output_dir / 'report.pdf'}")

    return report


def main() -> None:
    """Full pipeline: ingest seed data, train, generate reports."""
    project_root = Path(__file__).resolve().parent.parent
    seed_log = project_root / "data" / "seed_log.txt"
    database_path = project_root / "fco_ai_analyzer.sqlite3"
    reports_dir = project_root / "reports"

    print("=" * 60)
    print("FCO AI Analyzer - Full Pipeline")
    print("=" * 60)

    print("\n1️⃣  Seeding database...")
    seed_database(seed_log, database_path)

    print("\n2️⃣  Training models...")
    train_models(seed_log, database_path)

    print("\n3️⃣  Generating reports...")
    report = generate_reports(seed_log, database_path, reports_dir)

    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    summary = report["summary"]
    recommendation = report["recommendation"]
    patterns = report["patterns"]

    print(f"Total attempts: {summary.total_attempts}")
    print(f"Wins: {summary.wins} ({summary.win_rate:.1%})")
    print(f"Losses: {summary.losses}")
    print()
    print(f"RECOMMENDATION: {recommendation.recommendation}")
    print(f"Success probability: {recommendation.success_probability:.1%}")
    print(f"Confidence: {recommendation.confidence}")
    print()
    print("Top 5 Patterns:")
    for i, pattern in enumerate(patterns[:5], 1):
        print(
            f"  {i}. Pattern {pattern.pattern}: "
            f"n={pattern.sample_size}, "
            f"wins={pattern.wins}, "
            f"win_rate={pattern.win_rate:.1%}, "
            f"lift={pattern.lift:.2f}x"
        )
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
