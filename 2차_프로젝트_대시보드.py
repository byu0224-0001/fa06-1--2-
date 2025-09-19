import streamlit as st
import pandas as pd
import numpy as np
from rice_backend import get_item_history, predict_item_price, get_rice_history, predict_rice_price
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================================================================
# Streamlit 페이지 기본 설정
# ==============================================================================
st.set_page_config(
    page_title="사장님을 위한 AI 식자재 비서",
    page_icon="🧑‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# AI 구매 팁 함수
# ==============================================================================
def _add_ai_purchase_tip(item_name, history, prediction, predict_days):
    """AI 구매 팁을 추가하는 함수"""
    try:
        current_price = history['가격'].iloc[-1]
        
        # 최저가 찾기
        min_price = prediction['가격'].min()
        min_price_date = prediction[prediction['가격'] == min_price]['날짜'].iloc[0]
        min_price_date_str = min_price_date.strftime('%m월 %d일')
        
        # 가격 변화 분석
        price_trend = "하락" if prediction['가격'].iloc[-1] < prediction['가격'].iloc[0] else "상승"
        
        # 구매 권장 시점 결정
        if min_price < current_price * 0.95:  # 5% 이상 저렴한 시점이 있으면
            recommendation = f"{min_price_date_str}에 {int(min_price):,}원으로 가장 저렴할 것으로 예측됩니다."
            tip = f"{min_price_date_str}에 구매하여 비용을 절감할 것을 권장합니다."
        else:
            recommendation = f"분석 기간 동안 {item_name} 가격이 {price_trend}하는 추세를 보입니다."
            tip = "현재 가격이 적정 수준이므로 필요시 구매하시면 됩니다."
        
        # AI 구매 팁 표시
        with st.expander("🤖 AI 구매 팁", expanded=False):
            st.markdown(f"**결론:** {recommendation}")
            st.markdown(f"**분석:** 분석 기간 동안 {item_name} 가격이 {price_trend}하며, {min_price_date_str}에 가장 낮은 가격을 기록할 것으로 예상됩니다.")
            st.markdown(f"**팁:** {tip}")
            
    except Exception as e:
        st.error(f"AI 구매 팁 생성 중 오류가 발생했습니다: {e}")

# ==============================================================================
# 데이터 시뮬레이션 함수 (백엔드 API 및 DB 연동으로 대체될 부분)
# ==============================================================================
@st.cache_data
def load_and_prepare_data(item_name):
    """
    모든 품목에 대해 백엔드에서 실제 데이터를 로딩합니다.
    """
    try:
        data = get_item_history(item_name, days=365)
        if data.empty or len(data) == 0:
            raise ValueError("빈 데이터 반환")
        return data
    except Exception as e:
        print(f"{item_name} 데이터 로드 오류: {e}")
        # 품목별 기본 가격 설정
        base_prices = {
            "쌀": 52000, "감자": 3000, "배추": 2000, "양파": 18000, "오이": 4000,
            "상추": 1500, "무": 1000, "파": 2000, "건고추": 25000,
            "깐마늘(국산)": 8000, "깐마늘(수입)": 6000
        }
        base_price = base_prices.get(item_name, 5000)
        dates = pd.to_datetime(pd.date_range(end=datetime.today(), periods=365))
        prices = np.full(365, base_price)
        return pd.DataFrame({'날짜': dates, '가격': prices})

def generate_future_predictions_for_item(item_name, price_history, days_to_predict):
    # 모든 품목: 백엔드 예측 사용
    try:
        return predict_item_price(item_name, price_history, days_to_predict)
    except Exception as e:
        print(f"{item_name} 예측 오류: {e}")
        # 폴백: 간단한 트렌드 기반 예측
        last_date = price_history['날짜'].max()
        last_price = float(price_history['가격'].iloc[-1])
        trend = np.linspace(1.0, 1.0 + 0.10, days_to_predict)  # 최대 +10%
        future_dates = pd.to_datetime(pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict))
        future_prices = (last_price * trend).astype(float)
        return pd.DataFrame({'날짜': future_dates, '가격': future_prices})

# ==============================================================================
# 🧭 사이드바 UI: 페이지 네비게이션 메뉴 (DOCX 파일 기반)
# ==============================================================================
with st.sidebar:
    st.image("https://placehold.co/300x100/FFFFFF/333333?text=OUR+LOGO&font=Inter", width='stretch')
    # st.session_state를 이용하여 현재 페이지 상태를 저장하고, 버튼 클릭으로 변경합니다.
    if st.button("🏠 메인 대시보드", width='stretch'):
        st.session_state.page = "main"
        st.rerun() 
    
    if st.button("📊 원가 분석", width='stretch'):
        st.session_state.page = "cost_analysis"
        st.rerun()

    if st.button("💡 서비스 소개", width='stretch'):
        st.session_state.page = "about"
        st.rerun()

# ==============================================================================
# 🎈 메인 대시보드 페이지 함수
# ==============================================================================
def main_dashboard():
    st.title("🧑‍🍳 사장님을 위한 AI 식자재 비서")
    st.markdown(f"**기준일:** `{datetime.now().strftime('%Y-%m-%d')}` | **가게명:** `행복한 밥집`")
    st.divider()

    # --- 주요 품목 시세 요약 ---
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("주요 품목 가격 예측")
    with header_cols[1]:
        period_options = {0: "오늘 시세", 7: "1주일 예측", 15: "15일 예측", 30: "1개월 예측"}
        st.session_state.predict_days = st.selectbox("표시할 정보", options=list(period_options.keys()), format_func=lambda x: period_options[x], label_visibility="collapsed")

    summary_cols = st.columns(3)
    items = {"쌀": "🍚", "건고추": "🌶️", "양파": "🧅"}
    price_units = {"쌀": "20kg", "건고추": "600g", "양파": "15kg"}
    for item_name, emoji in items.items():
        with summary_cols.pop(0): # Use pop to iterate through columns
            history = load_and_prepare_data(item_name)
            # 데이터 검증 추가
            if history.empty or len(history) == 0:
                st.error(f"{item_name} 데이터를 로드할 수 없습니다.")
                continue
            current_price = history['가격'].iloc[-1]
            with st.container(border=True):
                st.markdown(f"<h5>{emoji} {item_name} ({price_units[item_name]})</h5>", unsafe_allow_html=True)
                if st.session_state.predict_days == 0:
                    yesterday_price = history['가격'].iloc[-2]
                    daily_change = current_price - yesterday_price
                    price_color = "#E84A5F" if daily_change > 0 else "#3182F6"
                    price_arrow = "▲" if daily_change > 0 else "▼"
                    change_text = "올랐어요!" if daily_change > 0 else "내렸어요."
                    st.markdown(f"<h2 style='display: inline;'>{int(current_price):,}원</h2> <span style='color:{price_color};'>{price_arrow} {int(abs(daily_change)):,}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top:0.5rem;'>어제보다 {int(abs(daily_change)):,}원 {change_text}</p>", unsafe_allow_html=True)
                else:
                    prediction = generate_future_predictions_for_item(item_name, history, st.session_state.predict_days)
                    predicted_price = prediction['가격'].iloc[-1]
                    future_change = predicted_price - current_price
                    price_color = "#E84A5F" if future_change > 0 else "#3182F6"
                    price_arrow = "▲" if future_change > 0 else "▼"
                    change_text = "오를 전망" if future_change > 0 else "내릴 전망"
                    st.markdown(f"<h2 style='display: inline;'>{int(predicted_price):,}원</h2> <span style='color:{price_color};'>{price_arrow} {int(abs(future_change)):,}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top:0.5rem;'>현재보다 {int(abs(future_change)):,}원 {change_text}이에요.</p>", unsafe_allow_html=True)
                    
                    # AI 구매 팁 추가
                    _add_ai_purchase_tip(item_name, history, prediction, st.session_state.predict_days)
                if st.button(f"상세 예측 보기", key=f"details_{item_name}", width='stretch'):
                    st.session_state.page, st.session_state.selected_item = 'detail', item_name
                    st.rerun()
    st.divider()

    # --- [개선] 재고 관리 섹션 확장 및 시각화 ---
    st.subheader("📦 주요 품목 재고 현황")
    inventory_data = {
        "쌀": {"icon": "🍚", "current": 2, "total": 10, "unit": "포대"},
        "건고추": {"icon": "🌶️", "current": 8, "total": 20, "unit": "봉지"},
        "양파": {"icon": "🧅", "current": 12, "total": 15, "unit": "망"}
    }
    inventory_cols = st.columns(3)
    low_stock_item = None
    for item_name, data in inventory_data.items():
        with inventory_cols.pop(0):
            percentage = (data['current'] / data['total']) * 100
            st.markdown(f"**{data['icon']} {item_name} 재고**")
            st.progress(int(percentage), text=f"{data['current']} / {data['total']} {data['unit']}")
            if percentage < 25: # 재고 25% 미만 시 경고 및 구매 추천 대상 선정
                st.warning(f"재고가 부족해요! ({int(percentage)}%)", icon="⚠️")
                if not low_stock_item: low_stock_item = item_name
    
    st.info("재고 현황은 수기 입력을 통해서도 업데이트할 수 있습니다.")
    st.divider()
    
    # --- 구매 추천 및 선도거래 ---
    if low_stock_item: # 재고 부족 품목이 있을 때만 구매 추천 표시
        st.subheader(f"🛒 부족한 {low_stock_item} 구매 추천")
        try:
            history_data = load_and_prepare_data(low_stock_item)
            if history_data.empty or len(history_data) == 0:
                st.error(f"{low_stock_item} 데이터를 로드할 수 없습니다.")
            else:
                today_price = history_data['가격'].iloc[-1]
                predictions = generate_future_predictions_for_item(low_stock_item, history_data, 14)
                if predictions.empty or len(predictions) == 0:
                    st.error(f"{low_stock_item} 예측 데이터를 생성할 수 없습니다.")
                else:
                    future_price_14d = predictions['가격'].iloc[-1]
                    price_diff = int(future_price_14d - today_price)
                    if price_diff > 0:
                        st.success(f"**지금 구매하세요!** AI 예측 결과, 2주 뒤보다 약 **{price_diff:,}원** 저렴합니다!", icon="👍")
                    else:
                        st.warning("**구매 보류.** 2주 내 가격이 안정적이거나 하락할 전망입니다.", icon="🤔")
        except Exception as e:
            st.error(f"구매 추천 데이터를 처리하는 중 오류가 발생했습니다: {str(e)}")
    
    if st.button("🌾 농산물 바로 구매하러 가기", width='stretch'):
        st.toast("식자재 구매 서비스 페이지로 이동합니다.(준비중이예요)")

# ==============================================================================
# 📊 원가 분석 페이지 함수
# ==============================================================================
def cost_analysis_page():
    st.title("📊 원가 분석")
    st.markdown("가게의 현재 실적을 확인하고, 다양한 시나리오를 시뮬레이션 해보세요.")
    st.divider()

    # 1. 현재 실적 요약 (메인 페이지에서 이동)
    st.subheader("💰 현재 실적 요약")
    metric_cols = st.columns(2)
    metric_cols[0].metric(label="당월 누적 매출", value="5,230,000 원", delta="어제 대비 2.5%")
    metric_cols[1].metric(label="당월 식자재 원가율", value="35.2 %", delta="-1.2%", delta_color="inverse")
    st.divider()
    
    # 2. 실시간 원가율 계산기
    st.subheader("🧮 실시간 마진율 계산기")
    calc_cols = st.columns(2)
    with calc_cols[0]:
        sales = st.number_input("월 목표 매출액 (원)", value=15000000, step=100000)
        food_cost = st.number_input("월 예상 식재료비 (원)", value=5250000, step=50000, help="AI 예측 기반 값이며, 수정 가능합니다.")
        labor_cost = st.number_input("월 인건비 (원)", value=3000000, step=100000)
        rent_cost = st.number_input("월 임대료 (원)", value=2000000, step=50000)
    with calc_cols[1]:
        st.write("") # 여백
        st.write("") # 여백
        if st.button("마진율 계산하기", width='stretch'):
            total_cost = food_cost + labor_cost + rent_cost
            operating_profit = sales - total_cost
            profit_margin = (operating_profit / sales) * 100 if sales > 0 else 0
            
            st.metric(label="예상 마진금액", value=f"{int(operating_profit):,} 원")
            st.metric(label="예상 마진율", value=f"{profit_margin:.2f} %")

# ==============================================================================
# 💡 서비스 소개 페이지 함수
# ==============================================================================
def about_service_page():
    st.title("💡 서비스 소개")
    st.markdown("저희 서비스는 데이터와 AI 기술을 통해 사장님의 성공을 돕습니다.")
    st.divider()

    st.subheader("AI 가격 예측 시스템이란?")
    st.write("과거 데이터와 다양한 변수를 분석하여 미래 식자재 가격을 예측합니다. 사장님의 합리적인 구매 결정을 도와 원가 절감에 기여합니다.")
    st.subheader("농산물 선도 거래 서비스란?")
    st.write("미래의 가격을 예측하여, 가격이 오르기 전에 더 저렴한 가격으로 식자재를 미리 구매(계약)할 수 있도록 돕는 서비스입니다.")
    st.divider()
    st.subheader("🤖 AI 경영 어드바이저 (출시 예정)")
    st.info("가게의 데이터를 기반으로 맞춤형 경영 컨설팅을 제공하는 AI 챗봇 서비스가 출시될 예정입니다. 기대해주세요!")

# ==============================================================================
# 📈 상세 예측 페이지 함수
# ==============================================================================
def detail_page():
    item_name = st.session_state.selected_item
    emoji = {"쌀": "🍚", "건고추": "🌶️", "양파": "🧅"}[item_name]
    unit = {"쌀": "20kg", "건고추": "600g", "양파": "15kg"}[item_name]

    st.title(f"{emoji} {item_name} 상세 가격 예측")
    if st.button("⬅️ 메인 대시보드로 돌아가기"):
        st.session_state.page = 'main'
        st.rerun()

    period_options = {7: "1주일", 15: "15일", 30: "1개월", 90: "3개월"}
    default_period = st.session_state.get('predict_days', 7)
    if default_period == 0: default_period = 7
    
    predict_days = st.radio(
        "예측 기간 선택", options=list(period_options.keys()),
        format_func=lambda x: period_options[x],
        index=list(period_options.keys()).index(default_period),
        horizontal=True
    )
    st.session_state.predict_days = predict_days

    # 모든 품목: 백엔드 연동으로 실제 데이터/예측 사용
    price_history = load_and_prepare_data(item_name)
    predictions = generate_future_predictions_for_item(item_name, price_history, predict_days)
    
    # 데이터 검증
    if price_history.empty or len(price_history) == 0:
        st.error(f"{item_name} 데이터를 로드할 수 없습니다.")
        return
    if predictions.empty or len(predictions) == 0:
        st.error(f"{item_name} 예측 데이터를 생성할 수 없습니다.")
        return
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_history['날짜'].tail(60), y=price_history['가격'].tail(60), mode='lines', name='과거 데이터', line=dict(color='darkgrey', width=2)))
    last_history_point = price_history.tail(1)
    connected_predictions = pd.concat([last_history_point, predictions], ignore_index=True)
    fig.add_trace(go.Scatter(x=connected_predictions['날짜'], y=connected_predictions['가격'], mode='lines', name='예측 데이터', line=dict(color='royalblue', width=3)))
    fig.update_layout(title=dict(text=f'{item_name} ({unit}) 가격 추이 및 예측', x=0.5), yaxis_title=f'가격 (원/{unit})', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
    st.plotly_chart(fig, width='stretch')

    # 간단한 요약 지표
    current_price = int(price_history['가격'].iloc[-1])
    future_price = int(predictions['가격'].iloc[-1])
    diff = future_price - current_price
    cols = st.columns(3)
    cols[0].metric(label="현재 가격", value=f"{current_price:,} 원")
    cols[1].metric(label=f"{period_options[predict_days]} 후 예측", value=f"{future_price:,} 원")
    cols[2].metric(label="변화", value=("+" if diff>=0 else "")+f"{diff:,} 원", delta=f"{diff:,} 원")

# ==============================================================================
# 🧭 페이지 라우팅 (Page Routing)
# ==============================================================================
if 'page' not in st.session_state: st.session_state.page = 'main'
if 'predict_days' not in st.session_state: st.session_state.predict_days = 0

# 페이지 선택에 따라 해당 함수를 호출합니다.
if st.session_state.page == 'main':
    main_dashboard()
elif st.session_state.page == 'cost_analysis':
    cost_analysis_page()
elif st.session_state.page == 'about':
    about_service_page()
elif st.session_state.page == 'detail':
    # placeholder 코드를 삭제하고 실제 함수를 호출합니다.
    detail_page()

# --- 페이지 전체 스타일링을 위한 CSS ---
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h5 { margin-bottom: 0.5rem; font-weight: 600; color: #333; }
    h2 { margin-bottom: 0.2rem; }
    div[data-testid="stMetric"], div[data-testid="stButton"] > button {
        background-color: #FFFFFF; border: 1px solid #E0E0E0;
        border-radius: 10px; padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

