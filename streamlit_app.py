from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from fco_ai_analyzer.history import HistoryManager
from fco_ai_analyzer.parser import LogParser
from fco_ai_analyzer.pipeline import FCOAnalyzer
from fco_ai_analyzer.reporting import ReportExporter


st.set_page_config(page_title="FCO AI Analyzer - Live Decision", layout="wide")
st.title("🎮 FCO AI Analyzer - Real-time Decision Engine")
st.caption("Hướng dẫn đập thẻ realtime dựa trên dữ liệu của bạn")

project_root = Path(__file__).resolve().parent
data_root = Path(os.getenv("DATA_DIR", project_root / "data"))
data_root.mkdir(parents=True, exist_ok=True)

analyzer = FCOAnalyzer(data_root / "fco_ai_analyzer.sqlite3")
exporter = ReportExporter()
history_manager = HistoryManager(
    history_path=data_root / "history_log.txt",
    seed_path=project_root / "data" / "seed_log.txt",
)

if "raw_log" not in st.session_state:
    st.session_state.raw_log = history_manager.read_all()
    st.session_state.parsed_log = LogParser().parse(st.session_state.raw_log)

parsed_log = st.session_state.parsed_log

with st.sidebar:
    st.subheader("Lich su dap the")
    st.caption("Them session moi de tang du lieu va cap nhat ti le")
    new_session_text = st.text_area(
        "Session log moi",
        value="",
        placeholder="Vi du:\n2 1 = 4\n= 2\n1 = 4",
        height=180,
    )
    if st.button("Luu session va cap nhat", use_container_width=True):
        cleaned = new_session_text.strip()
        if not cleaned:
            st.warning("Vui long nhap session log truoc khi luu")
        else:
            history_manager.append_session(cleaned)
            was_ingested = analyzer.ingest_text(cleaned, source_name="streamlit_sidebar")
            st.session_state.raw_log = history_manager.read_all()
            st.session_state.parsed_log = LogParser().parse(st.session_state.raw_log)
            st.success("Da luu session moi vao lich su")
            if not was_ingested:
                st.info("Session nay da tung duoc nap truoc do, he thong bo qua ban ghi trung")
    if st.button("Undo session cuoi", use_container_width=True):
        removed = history_manager.pop_last_session()
        if removed is None:
            st.warning("Khong co session nao de xoa")
        else:
            refreshed_text = history_manager.read_all()
            analyzer.rebuild_from_history(refreshed_text, source_name="undo_rebuild")
            st.session_state.raw_log = refreshed_text
            st.session_state.parsed_log = LogParser().parse(refreshed_text)
            st.success("Da xoa session cuoi va rebuild lai du lieu")
    st.divider()
    st.metric("Tong so ky tu log", len(st.session_state.raw_log))
    st.metric("Tong session", history_manager.session_count())
    st.metric("Tong lan dap chinh", len(st.session_state.parsed_log.attempts))

tab1, tab2, tab3 = st.tabs(["🎯 Decision Now", "📊 Analytics", "📁 Export"])

with tab1:
    st.subheader("⚡ Nhập chuỗi mồi hiện tại")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        pattern_input = st.text_input(
            "Chuỗi mồi (ví dụ: 2,1 hoặc 4 hoặc để trống để đập ngay)",
            value="",
            placeholder="2,1 hoặc 4 hoặc để trống"
        )
    
    if st.button("🔮 Phân tích ngay", use_container_width=True, type="primary"):
        try:
            if pattern_input.strip():
                current_pattern = tuple(int(token.strip()) for token in pattern_input.split(","))
            else:
                current_pattern = tuple()
            
            recommendation = analyzer.recommendation_engine.recommend(parsed_log, current_pattern=current_pattern, model_bundle=analyzer.model_bundle)
            
            st.success("✅ Phân tích xong!")
            
            if recommendation.recommendation == "ĐẬP":
                st.markdown(
                    f"""<div style="background-color: #28a745; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                    <h1 style="color: white; margin: 0;">🔥 ĐẬP 🔥</h1>
                    <p style="color: white; font-size: 18px; margin: 10px 0;">Xác suất thắng: <strong>{recommendation.success_probability:.1%}</strong></p>
                    <p style="color: white; font-size: 14px; margin: 0;">Độ tin cậy: {recommendation.confidence}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="background-color: #ffc107; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                    <h1 style="color: black; margin: 0;">⏸️ WAIT ⏸️</h1>
                    <p style="color: black; font-size: 18px; margin: 10px 0;">Xác suất thắng: <strong>{recommendation.success_probability:.1%}</strong></p>
                    <p style="color: black; font-size: 14px; margin: 0;">Độ tin cậy: {recommendation.confidence}</p>
                    </div>""",
                    unsafe_allow_html=True
                )
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sample Size", recommendation.sample_size)
            col2.metric("Win Rate", f"{recommendation.win_rate:.1%}")
            col3.metric("Bayesian", f"{recommendation.bayesian_probability:.1%}")
            col4.metric("Markov", f"{recommendation.markov_probability:.1%}")
            
            st.subheader("💭 Giải thích chi tiết:")
            for line in recommendation.rationale:
                st.write(f"• {line}")
        
        except ValueError as e:
            st.error(f"❌ Lỗi: {e}")

with tab2:
    st.subheader("📈 Toàn bộ dữ liệu phân tích")
    
    report = analyzer.build_report(st.session_state.raw_log)
    summary = report["summary"]
    patterns = report["patterns"]
    recommendation = report["recommendation"]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Attempts", summary.total_attempts)
    col2.metric("Wins", summary.wins)
    col3.metric("Losses", summary.losses)
    col4.metric("Win Rate", f"{summary.win_rate:.1%}")
    
    st.subheader("🏆 Top 20 Best Patterns")
    st.dataframe(
        [
            {
                "Pattern": " → ".join(str(token) for token in item.pattern) if item.pattern else "(đập ngay)",
                "Samples": item.sample_size,
                "Wins": item.wins,
                "Win Rate": f"{item.win_rate:.1%}",
                "Lift": f"{item.lift:.2f}x",
            }
            for item in patterns[:20]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🕒 Timeline theo session")
    timeline_rows: list[dict[str, float | int]] = []
    cumulative_attempts = 0
    cumulative_wins = 0
    for session in parsed_log.sessions:
        attempts = len(session.attempts)
        wins = sum(1 for item in session.attempts if item.is_win)
        losses = attempts - wins
        session_win_rate = (wins / attempts) if attempts else 0.0
        cumulative_attempts += attempts
        cumulative_wins += wins
        cumulative_win_rate = (cumulative_wins / cumulative_attempts) if cumulative_attempts else 0.0
        timeline_rows.append(
            {
                "session_index": session.session_index,
                "attempts": attempts,
                "wins": wins,
                "losses": losses,
                "session_win_rate": session_win_rate,
                "cumulative_win_rate": cumulative_win_rate,
            }
        )

    if timeline_rows:
        timeline_df = pd.DataFrame(timeline_rows)
        st.line_chart(
            timeline_df.set_index("session_index")[["session_win_rate", "cumulative_win_rate"]],
            use_container_width=True,
        )
        st.dataframe(
            timeline_df,
            use_container_width=True,
            hide_index=True,
        )

with tab3:
    st.subheader("📥 Xuất báo cáo")
    
    report = analyzer.build_report(st.session_state.raw_log)
    summary = report["summary"]
    patterns = report["patterns"]
    recommendation = report["recommendation"]
    
    if st.button("📄 Xuất HTML"):
        output_path = data_root / "analysis_latest.html"
        exporter.export_html(output_path, summary, patterns, recommendation)
        st.success(f"✅ HTML: {output_path}")
    
    if st.button("📊 Xuất Excel"):
        output_path = data_root / "analysis_latest.xlsx"
        exporter.export_excel(output_path, summary, patterns, recommendation)
        st.success(f"✅ Excel: {output_path}")
    
    if st.button("📕 Xuất PDF"):
        output_path = data_root / "analysis_latest.pdf"
        exporter.export_pdf(output_path, summary, patterns, recommendation)
        st.success(f"✅ PDF: {output_path}")

