import streamlit as st
from config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title("🏦 HCIS 신용평가 시스템")

st.markdown("""
안녕하세요. Home Credit을 찾아주셔서 감사합니다.

본 시스템은 Home Credit 데이터를 기반으로 대출 심사 프로세스를 운영 관점에서 재구성한 신용평가 데모입니다.

고객 단위 조회를 통해 **PD → 점수 → 등급 → 최종 심사결정 → 행동 추천**까지의 흐름을 일관된 기준으로 확인할 수 있습니다.

좌측 상단(페이지 메뉴)에서 페이지를 이동해 주십시오.
""")