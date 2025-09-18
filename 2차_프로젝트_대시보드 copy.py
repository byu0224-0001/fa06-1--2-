import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================================================================
# Streamlit 페이지 기본 설정
# ==============================================================================
st.set_page_config(
    page_title="사장님을 위한 AI 식자재 비서",
    page_icon="🧑‍🍳",
    layout="wide",
    initial_sidebar_state="expanded" # 사이드바를 기본적으로 열어둠
)

# ==============================================================================
# 데이터 시뮬레이션 함수 (백엔드 API 및 DB 연동으로 대체될 부분)
# ==============================================================================
@st.cache_data
def load_and_prepare_data(item_name):
    """
    초기 데이터를 로딩하고, 품목별로 가격대를 다르게 시뮬레이션합니다.
    """
    base_price = 52000
    if item_name == "건고추": base_price = 25000
    elif item_name == "양파": base_price = 18000

    try:
        # 인코딩 불일치에 대비한 다중 시도
        encodings_to_try = ['utf-8-sig', 'utf-8', 'cp949', 'ISO-8859-1']
        df = None
        last_error = None
        
        for enc in encodings_to_try:
            try:
                df = pd.read_csv('rice.csv', encoding=enc, encoding_errors='replace')
                # 컬럼 확인 후 성공시 break
                if '날짜' in df.columns and '가격(20kg)' in df.columns:
                    break
                else:
                    print(f"인코딩 {enc}로 읽었지만 필요한 컬럼이 없습니다. 컬럼: {list(df.columns)}")
                    continue
            except Exception as e:
                last_error = e
                print(f"인코딩 {enc} 시도 실패: {e}")
                continue
        
        if df is None or '날짜' not in df.columns:
            raise Exception(f"rice.csv 파일을 읽을 수 없습니다. 마지막 오류: {last_error}")
        
        df['날짜'] = pd.to_datetime(df['날짜'])
        price_history = df.groupby('날짜')['가격(20kg)'].mean().reset_index()
        price_history = price_history.sort_values('날짜').tail(365)
        price_history.rename(columns={'가격(20kg)': '가격'}, inplace=True)
        price_history['가격'] = price_history['가격'] / price_history['가격'].mean() * base_price
        return price_history
    except FileNotFoundError:
        dates = pd.to_datetime(pd.date_range(end=datetime.today(), periods=365))
        prices = np.random.normal(loc=base_price, scale=base_price*0.1, size=365)
        return pd.DataFrame({'날짜': dates, '가격': prices})

def generate_future_predictions(price_history, days_to_predict):
    last_date = price_history['날짜'].max()
    last_price = price_history['가격'].iloc[-1]
    future_dates = pd.to_datetime(pd.date_range(start=last_date + timedelta(days=1), periods=days_to_predict))
    trend_factor = np.linspace(1, 1 + np.random.uniform(-0.15, 0.15), days_to_predict)
    noise = np.random.normal(0, last_price * 0.02, days_to_predict)
    future_prices = last_price * trend_factor + noise
    return pd.DataFrame({'날짜': future_dates, '가격': future_prices.astype(int)})

# ==============================================================================
# 🧭 사이드바 UI 구성 (DOCX 파일 기반)
# ==============================================================================
with st.sidebar:
    st.image("https://placehold.co/300x100/FFFFFF/333333?text=OUR+LOGO&font=Inter", use_column_width=True)
    st.title("경영 분석 도구")
    st.write("가게의 재무 상태를 분석하고 미래를 계획하세요.")
    st.divider()

    # --- 원가율 계산기 ---
    st.subheader("📊 실시간 원가율 계산기")
    
    # st.number_input을 사용하여 사장님이 직접 값을 입력하고 수정할 수 있게 합니다.
    sales = st.number_input("월 목표 매출액 (원)", value=15000000, step=100000)
    # 식재료비는 AI 예측을 기반으로 하되, 사장님이 직접 수정할 수 있는 옵션을 제공하는 것이 좋습니다.
    food_cost = st.number_input("월 예상 식재료비 (원)", value=5250000, step=50000, help="AI가 예측한 평균 식재료비입니다. 직접 수정할 수 있습니다.")
    labor_cost = st.number_input("월 인건비 (원)", value=3000000, step=100000)
    rent_cost = st.number_input("월 임대료 (원)", value=2000000, step=50000)
    
    if st.button("마진 계산하기", use_container_width=True):
        total_cost = food_cost + labor_cost + rent_cost
        operating_profit = sales - total_cost
        profit_margin = (operating_profit / sales) * 100 if sales > 0 else 0
        
        st.success(f"예상 마진금액: **{int(operating_profit):,} 원**")
        st.success(f"예상 마진율: **{profit_margin:.2f} %**")

    st.divider()

    # --- 서비스 소개 ---
    st.subheader("💡 서비스 소개")
    with st.expander("AI 가격 예측 시스템이란?"):
        st.write("과거 데이터와 다양한 변수를 분석하여 미래 식자재 가격을 예측합니다. 사장님의 합리적인 구매 결정을 도와 원가 절감에 기여합니다.")
    with st.expander("농산물 선도 거래 서비스란?"):
        st.write("미래의 가격을 예측하여, 가격이 오르기 전에 더 저렴한 가격으로 식자재를 미리 구매(계약)할 수 있도록 돕는 서비스입니다.")

    st.divider()
    
    # --- 챗봇 도입 (미래 비전) ---
    st.subheader("🤖 AI 경영 어드바이저 (출시 예정)")
    st.info("가게의 데이터를 기반으로 맞춤형 경영 컨설팅을 제공하는 AI 챗봇 서비스가 출시될 예정입니다. 기대해주세요!")


# ==============================================================================
# 🎈 메인 대시보드 페이지 함수
# ==============================================================================
def main_dashboard():
    st.title("🧑‍🍳 사장님을 위한 AI 식자재 비서")
    st.markdown(f"**기준일:** `{datetime.now().strftime('%Y-%m-%d')}` | **가게명:** `행복한 밥집`")
    st.divider()

    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("주요 품목 시세 및 예측")
    with header_cols[1]:
        period_options = {
            0: "오늘 시세", 7: "1주일 예측", 15: "15일 예측",
            30: "1개월 예측", 90: "3개월 예측"
        }
        st.session_state.predict_days = st.selectbox(
            "표시할 정보 선택",
            options=list(period_options.keys()),
            format_func=lambda x: period_options[x],
            index=0,
            label_visibility="collapsed"
        )

    summary_cols = st.columns(3)
    items = {"쌀": "🍚", "건고추": "🌶️", "양파": "🧅"}
    price_units = {"쌀": "20kg", "건고추": "600g", "양파": "15kg"}

    for i, (item_name, emoji) in enumerate(items.items()):
        with summary_cols[i]:
            history = load_and_prepare_data(item_name)
            current_price = history['가격'].iloc[-1]
            
            with st.container(border=True):
                st.markdown(f"<h5>{emoji} {item_name} ({price_units[item_name]})</h5>", unsafe_allow_html=True)
                
                if st.session_state.predict_days == 0:
                    yesterday_price = history['가격'].iloc[-2]
                    daily_change = current_price - yesterday_price
                    price_color = "#E84A5F" if daily_change > 0 else "#3182F6"
                    price_arrow = "▲" if daily_change > 0 else "▼"
                    change_text = "올랐어요!" if daily_change > 0 else "내렸어요."
                    st.markdown(f"<h2 style='display: inline;'>{int(current_price):,}원</h2> <span style='color:{price_color}; font-size:1.1em;'>{price_arrow} {int(abs(daily_change)):,}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top:0.5rem;'>어제보다 {int(abs(daily_change)):,}원 {change_text}</p>", unsafe_allow_html=True)
                else:
                    prediction = generate_future_predictions(history, st.session_state.predict_days)
                    predicted_price = prediction['가격'].iloc[-1]
                    future_change = predicted_price - current_price
                    price_color = "#E84A5F" if future_change > 0 else "#3182F6"
                    price_arrow = "▲" if future_change > 0 else "▼"
                    change_text = "오를 전망" if future_change > 0 else "내릴 전망"
                    st.markdown(f"<h2 style='display: inline;'>{int(predicted_price):,}원</h2> <span style='color:{price_color}; font-size:1.1em;'>{price_arrow} {int(abs(future_change)):,}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p style='margin-top:0.5rem;'>현재보다 {int(abs(future_change)):,}원 {change_text}이에요.</p>", unsafe_allow_html=True)

                if st.button(f"상세 예측 보기", key=f"details_{item_name}", use_container_width=True):
                    st.session_state.page = 'detail'
                    st.session_state.selected_item = item_name
                    st.rerun()

    st.divider()

    # --- 하단 2분할 레이아웃 ---
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("📦 재고 관리 및 구매 추천")
        st.info("현재 가게의 **쌀 재고**가 **2포대** 남았어요!", icon="🍚")
        today_price = load_and_prepare_data("쌀")['가격'].iloc[-1]
        future_price_14d = generate_future_predictions(load_and_prepare_data("쌀"), 14)['가격'].iloc[-1]
        price_diff = int(future_price_14d - today_price)
        if price_diff > 0:
            st.success(f"**구매 추천!** 지금 구매하시면 2주 뒤보다 약 **{price_diff:,}원** 저렴해요!", icon="👍")
        else:
            st.warning("**구매 보류.** 2주 내 가격이 안정적이거나 하락할 전망이에요.", icon="🤔")
        if st.button("🌾 농산물 바로 구매하러 가기", use_container_width=True):
            st.toast("식자재 구매 서비스 페이지로 이동합니다.(준비중이예요)")

    with col2:
        st.subheader("💰 현재 실적 요약")
        st.metric(label="당월 누적 매출", value="5,230,000 원", delta="어제 대비 2.5%")
        st.metric(label="당월 식자재 원가율", value="35.2 %", delta="-1.2%", delta_color="inverse")
        if st.button("🌾 농산물 바로 구매하러 가기", use_container_width=True):
            st.toast("식자재 구매 서비스 페이지로 이동합니다.(준비중이예요)")


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

    price_history = load_and_prepare_data(item_name)
    predictions = generate_future_predictions(price_history, predict_days)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_history['날짜'].tail(60), y=price_history['가격'].tail(60), mode='lines', name='과거 데이터', line=dict(color='darkgrey', width=2)))
    last_history_point = price_history.tail(1)
    connected_predictions = pd.concat([last_history_point, predictions], ignore_index=True)
    fig.add_trace(go.Scatter(x=connected_predictions['날짜'], y=connected_predictions['가격'], mode='lines', name='예측 데이터', line=dict(color='royalblue', width=3)))
    fig.update_layout(title=dict(text=f'{item_name} ({unit}) 가격 추이 및 예측', x=0.5), yaxis_title=f'가격 (원/{unit})', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 🧭 페이지 라우팅 (Page Routing)
# ==============================================================================
if 'page' not in st.session_state: st.session_state.page = 'main'
if 'predict_days' not in st.session_state: st.session_state.predict_days = 0

if st.session_state.page == 'main':
    main_dashboard()
elif st.session_state.page == 'detail':
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

