import streamlit as st

def render_underwriter_report(under: dict, band: str, score: float, margin: float):
    # 상단 한 줄 결론
    headline = under.get("headline") or under.get("summary") or ""
    risk_level = under.get("risk_level") or band

    st.markdown("### 🧾 심사 요약")
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #eee;border-radius:16px;padding:16px 18px;
                    box-shadow:0 4px 12px rgba(0,0,0,0.06);color:#111;margin-bottom:18px;">
          <div style="font-size:14px;color:#555;margin-bottom:6px;">결론</div>
          <div style="font-size:18px;font-weight:700;margin-bottom:10px;">
            {risk_level} · HCIS {score:.0f} · 마진 {margin:+.1f}
          </div>
          <div style="font-size:15px;line-height:1.55;color:#222;">
            {headline}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 핵심 드라이버(Top 3~5)
    drivers = under.get("key_drivers") or under.get("reason_contributions") or []
    if isinstance(drivers, list) and drivers:
        st.markdown("### 🔎 핵심 위험 요인 (요약)")
        for d in drivers[:5]:
            st.markdown(f"- {d}")

    # 리스크 완화 요인(있으면)
    mitigants = under.get("mitigants") or under.get("positive_factors") or []
    if isinstance(mitigants, list) and mitigants:
        st.markdown("### 🟢 완화 요인")
        for m in mitigants[:3]:
            st.markdown(f"- {m}")

    # 액션 아이템(심사팀이 바로 할 일)
    actions = under.get("next_actions") or under.get("recommended_actions") or []
    if isinstance(actions, list) and actions:
        st.markdown("### ✅ 다음 액션 (심사팀 체크리스트)")
        for a in actions[:6]:
            st.checkbox(a, value=False)

    # 확인 질문(필수 확인)
    questions = under.get("verification_questions") or under.get("questions") or []
    if isinstance(questions, list) and questions:
        st.markdown("### ❓ 추가 확인 질문")
        for q in questions[:6]:
            st.markdown(f"- {q}")
