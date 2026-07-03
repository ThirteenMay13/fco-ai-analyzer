from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from jinja2 import Template

try:  # pragma: no cover - optional dependency guard
    from openpyxl import Workbook
except Exception:  # pragma: no cover - optional dependency guard
    Workbook = None

try:  # pragma: no cover - optional dependency guard
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - optional dependency guard
    A4 = None
    canvas = None

from .patterns import PatternStat
from .prediction import RecommendationResult
from .statistics import SummaryStatistics


class ReportExporter:
    """Export analysis artifacts to Excel, HTML, and PDF."""

    def export_excel(
        self,
        output_path: Path,
        summary: SummaryStatistics,
        patterns: list[PatternStat],
        recommendation: RecommendationResult,
    ) -> None:
        if Workbook is None:
            raise RuntimeError("openpyxl is required for Excel export")
        workbook = Workbook()
        summary_sheet = workbook.active
        summary_sheet.title = "Summary"
        summary_sheet.append(["metric", "value"])
        for key, value in asdict(summary).items():
            summary_sheet.append([key, value])
        summary_sheet.append(["recommendation", recommendation.recommendation])
        summary_sheet.append(["success_probability", recommendation.success_probability])
        summary_sheet.append(["confidence", recommendation.confidence])

        pattern_sheet = workbook.create_sheet("Patterns")
        pattern_sheet.append(["pattern", "sample_size", "wins", "losses", "win_rate", "confidence", "lift", "support"])
        for pattern in patterns:
            pattern_sheet.append([
                " ".join(str(token) for token in pattern.pattern),
                pattern.sample_size,
                pattern.wins,
                pattern.losses,
                pattern.win_rate,
                pattern.confidence,
                pattern.lift,
                pattern.support,
            ])
        workbook.save(output_path)

    def export_html(
        self,
        output_path: Path,
        summary: SummaryStatistics,
        patterns: list[PatternStat],
        recommendation: RecommendationResult,
    ) -> None:
        template = Template(
            """
            <html>
              <head><meta charset="utf-8"><title>FCO AI Analyzer Report</title></head>
              <body>
                <h1>FCO AI Analyzer Report</h1>
                <h2>Summary</h2>
                <ul>
                  <li>Total attempts: {{ summary.total_attempts }}</li>
                  <li>Wins: {{ summary.wins }}</li>
                  <li>Losses: {{ summary.losses }}</li>
                  <li>Win rate: {{ '%.2f'|format(summary.win_rate * 100) }}%</li>
                </ul>
                <h2>Recommendation</h2>
                <p><strong>{{ recommendation.recommendation }}</strong> | Probability: {{ '%.2f'|format(recommendation.success_probability * 100) }}% | Confidence: {{ recommendation.confidence }}</p>
                <h2>Top Patterns</h2>
                <table border="1" cellspacing="0" cellpadding="4">
                  <tr><th>Pattern</th><th>Samples</th><th>Win Rate</th><th>Lift</th></tr>
                  {% for item in patterns[:25] %}
                  <tr>
                    <td>{{ item.pattern }}</td>
                    <td>{{ item.sample_size }}</td>
                    <td>{{ '%.2f'|format(item.win_rate * 100) }}%</td>
                    <td>{{ '%.2f'|format(item.lift) }}x</td>
                  </tr>
                  {% endfor %}
                </table>
              </body>
            </html>
            """
        )
        output_path.write_text(template.render(summary=summary, patterns=patterns, recommendation=recommendation), encoding="utf-8")

    def export_pdf(
        self,
        output_path: Path,
        summary: SummaryStatistics,
        patterns: list[PatternStat],
        recommendation: RecommendationResult,
    ) -> None:
        if canvas is None or A4 is None:
            raise RuntimeError("reportlab is required for PDF export")
        pdf = canvas.Canvas(str(output_path), pagesize=A4)
        width, height = A4
        y_position = height - 40
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(40, y_position, "FCO AI Analyzer Report")
        y_position -= 30
        pdf.setFont("Helvetica", 11)
        for line in [
            f"Attempts: {summary.total_attempts}",
            f"Wins: {summary.wins}",
            f"Losses: {summary.losses}",
            f"Win rate: {summary.win_rate:.2%}",
            f"Recommendation: {recommendation.recommendation}",
            f"Success probability: {recommendation.success_probability:.2%}",
            f"Confidence: {recommendation.confidence}",
        ]:
            pdf.drawString(40, y_position, line)
            y_position -= 16
        y_position -= 10
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y_position, "Top Patterns")
        y_position -= 18
        pdf.setFont("Helvetica", 10)
        for pattern in patterns[:20]:
            pdf.drawString(
                40,
                y_position,
                f"{pattern.pattern} | samples={pattern.sample_size} | win_rate={pattern.win_rate:.2%} | lift={pattern.lift:.2f}x",
            )
            y_position -= 14
            if y_position < 60:
                pdf.showPage()
                y_position = height - 40
                pdf.setFont("Helvetica", 10)
        pdf.save()

