import streamlit as st
import pandas as pd
import numpy as np
from rice_backend import get_item_history, predict_item_price, get_rice_history, predict_rice_price
import plotly.graph_objects as go
from datetime import datetime, timedelta
import openai

# ==============================================================================
# Streamlit 페이지 기본 설정
# ==============================================================================
st.set_page_config(
    page_title="사장님을 위한 AI 식자재 비서",
    page_icon="🧑‍🍳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# AI 구매 팁 함수
# ==============================================================================
@st.cache_data(ttl=3600)
def generate_purchase_timing_report(df: pd.DataFrame, item_name: str, period_days: int):
    """가격 예측 데이터를 기반으로 최적 구매 시점을 분석하는 LLM 리포트를 생성합니다."""
    try:
        # OpenAI 클라이언트 초기화 (최신 방식)
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        # 날짜 포맷을 'YYYY-MM-DD'로 변경하여 LLM에 전달
        df_report = df.copy()
        df_report['날짜'] = pd.to_datetime(df_report['날짜']).dt.strftime('%Y-%m-%d')
        report_data_string = df_report.to_string(index=False)
        
        prompt = f"""
        당신은 식당 사장님을 위한 농산물 가격 분석 전문가입니다.
        주어진 '{item_name}'의 미래 가격 예측 데이터를 분석하여, 언제 구매하는 것이 가장 저렴할지 명확하게 알려주세요.

        [분석 대상]
        - 품목: {item_name}
        - 분석 기간: 앞으로 {period_days}일

        [가격 예측 데이터]
        {report_data_string}

        [리포트 작성 가이드]
        1. **결론**: 분석 기간 중 가장 가격이 저렴한 날짜와 예상 가격을 첫 문장에 명시해주세요. "결론적으로, {item_name}은(는) O월 O일에 OOO원으로 가장 저렴할 것으로 예측됩니다." 와 같은 형식으로 작성해주세요.
        2. **분석**: 전체적인 가격 변동 추세를 간략히 설명하고, 왜 해당 날짜가 최적의 구매 시점인지 덧붙여 설명해주세요.
        3. **팁**: 분석 결과를 바탕으로 사장님이 참고할 만한 간단한 팁을 한 문장으로 제안해주세요.
        4. 말투는 친절하고 단정적인 전문가 톤을 유지하고, 전체 내용을 3~4문장으로 요약해주세요.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 농산물 가격 예측 및 분석 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "authentication" in error_msg:
            return "❌ OpenAI API 키가 설정되지 않았거나 유효하지 않습니다. Streamlit Cloud의 Secrets 설정을 확인해주세요."
        elif "quota" in error_msg or "billing" in error_msg:
            return "❌ OpenAI API 사용량 한도를 초과했거나 결제 정보를 확인해주세요."
        elif "model" in error_msg:
            return "❌ 요청한 모델(gpt-4o)을 사용할 수 없습니다. 계정 권한을 확인해주세요."
        else:
            return f"❌ AI 구매 팁 생성 중 오류가 발생했습니다: {e}"

def _add_ai_purchase_tip(item_name, history, prediction, predict_days):
    """AI 구매 팁을 추가하는 함수 (LLM 기반)"""
    try:
        # 디버깅 정보 표시 (Streamlit Cloud에서만)
        if "streamlit.app" in st.get_option("server.baseUrlPath") or "share.streamlit.io" in st.get_option("server.baseUrlPath"):
            with st.expander("🔧 디버깅 정보", expanded=False):
                # Secrets 확인
                try:
                    api_key = st.secrets.get("OPENAI_API_KEY", "Not found")
                    if api_key != "Not found":
                        masked_key = api_key[:10] + "..." + api_key[-10:] if len(api_key) > 20 else "Too short"
                        st.write(f"API 키 상태: ✅ 설정됨 ({masked_key})")
                    else:
                        st.write("API 키 상태: ❌ 설정되지 않음")
                except Exception as e:
                    st.write(f"API 키 확인 오류: {e}")
        
        # LLM 기반 구매 타이밍 분석 리포트 생성
        report = generate_purchase_timing_report(prediction, item_name, predict_days)
        
        # AI 구매 팁 표시
        with st.expander("🤖 AI 구매 팁", expanded=False):
            st.markdown(report)
            
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

@st.cache_data
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
    # 상단 여백 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.title("🧑‍🍳 사장님을 위한 AI 식자재 비서")
    st.markdown(f"**기준일:** `2025-09-12` | **가게명:** `삼정캐대박맛집`")
    st.divider()

    # --- 주요 품목 시세 요약 ---
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("주요 품목 가격 예측")
    with header_cols[1]:
        period_options = {0: "오늘 시세", 3: "3일 예측", 7: "7일 예측", 15: "15일 예측"}
        st.session_state.predict_days = st.selectbox("표시할 정보", options=list(period_options.keys()), format_func=lambda x: period_options[x], label_visibility="collapsed", index=0)

    summary_cols = st.columns(3)
    items = {"쌀": "🍚", "깐마늘": "🧄", "양파": "🧅"}
    price_units = {"쌀": "20kg", "깐마늘": "20kg", "양파": "15kg"}
    for item_name, emoji in items.items():
        with summary_cols.pop(0): # Use pop to iterate through columns
            history = load_and_prepare_data(item_name)
            # 데이터 검증 추가
            if history.empty or len(history) == 0:
                st.error(f"{item_name} 데이터를 로드할 수 없습니다.")
                continue
            
            # 데이터 길이 검증 (최소 2개 행 필요)
            if len(history) < 2:
                st.error(f"{item_name} 데이터가 부족합니다. (현재: {len(history)}개 행)")
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
                    
                    # 오늘 시세일 때도 AI 구매 팁 표시 (3일 예측 기반)
                    try:
                        prediction = generate_future_predictions_for_item(item_name, history, 3)
                        _add_ai_purchase_tip(item_name, history, prediction, 3)
                    except Exception as e:
                        st.error(f"{item_name} AI 구매 팁 생성 중 오류 발생: {str(e)}")
                        continue
                else:
                    try:
                        prediction = generate_future_predictions_for_item(item_name, history, st.session_state.predict_days)
                        if prediction.empty or len(prediction) == 0:
                            st.error(f"{item_name} 예측 데이터를 생성할 수 없습니다.")
                            continue
                        predicted_price = prediction['가격'].iloc[-1]
                        future_change = predicted_price - current_price
                        price_color = "#E84A5F" if future_change > 0 else "#3182F6"
                        price_arrow = "▲" if future_change > 0 else "▼"
                        change_text = "오를 전망" if future_change > 0 else "내릴 전망"
                        st.markdown(f"<h2 style='display: inline;'>{int(predicted_price):,}원</h2> <span style='color:{price_color};'>{price_arrow} {int(abs(future_change)):,}</span>", unsafe_allow_html=True)
                        st.markdown(f"<p style='margin-top:0.5rem;'>현재보다 {int(abs(future_change)):,}원 {change_text}이에요.</p>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"{item_name} 예측 중 오류 발생: {str(e)}")
                        continue
                    
                    # AI 구매 팁 추가
                    _add_ai_purchase_tip(item_name, history, prediction, st.session_state.predict_days)
                if st.button(f"상세 예측 보기", key=f"details_{item_name}", width='stretch'):
                    st.session_state.page, st.session_state.selected_item = 'detail', item_name
                    st.rerun()
    st.divider()

    # --- 식자재 구매 섹션 ---
    st.subheader("🛒 식자재 구매")
    
    # 구매 옵션 버튼들
    purchase_cols = st.columns(2)
    
    with purchase_cols[0]:
        if st.button("🛒 식자재 바로 구매하기", width='stretch', use_container_width=True):
            st.session_state.page = 'purchase'
            st.rerun()
    
    with purchase_cols[1]:
        if st.button("📅 식자재 예약 구매하기", width='stretch', use_container_width=True):
            st.session_state.page = 'reservation'
            st.rerun()
    
    st.info("💡 AI 구매 팁을 참고하여 최적의 구매 시점을 선택하세요!")

# ==============================================================================
# 📊 원가 분석 페이지 함수
# ==============================================================================
def cost_analysis_page():
    # 상단 여백 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
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
    # 상단 여백 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
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
    # 상단 여백 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
    item_name = st.session_state.selected_item
    emoji_map = {"쌀": "🍚", "대파": "🧄", "양파": "🧅"}
    unit_map = {"쌀": "20kg", "대파": "20kg", "양파": "15kg"}
    emoji = emoji_map.get(item_name, "📦")
    unit = unit_map.get(item_name, "1kg")

    st.title(f"{emoji} {item_name} 상세 가격 예측")
    if st.button("⬅️ 메인 대시보드로 돌아가기"):
        st.session_state.page = 'main'
        st.rerun()

    period_options = {3: "3일", 7: "7일", 15: "15일"}
    default_period = st.session_state.get('predict_days', 3)
    if default_period == 0: default_period = 3
    
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
    if len(price_history) < 2:
        st.error(f"{item_name} 데이터가 부족합니다. (현재: {len(price_history)}개 행)")
        return
    if predictions.empty or len(predictions) == 0:
        st.error(f"{item_name} 예측 데이터를 생성할 수 없습니다.")
        return
    
    # MVP 가정: 9월 12일까지만 실제 데이터, 9월 13일부터는 예측 데이터로 표시
    cutoff_date = pd.Timestamp('2025-09-12')
    
    # 9월 12일까지의 실제 데이터 (과거 데이터 - 회색)
    historical_data = price_history[price_history['날짜'] <= cutoff_date]
    
    # 9월 13일부터의 실제 데이터 (예측 데이터로 표시 - 파란색)
    future_actual_data = price_history[price_history['날짜'] > cutoff_date]
    
    fig = go.Figure()
    
    # 과거 데이터 (9월 12일까지) - 회색
    if not historical_data.empty:
        fig.add_trace(go.Scatter(
            x=historical_data['날짜'].tail(60), 
            y=historical_data['가격'].tail(60), 
            mode='lines', 
            name='과거 데이터', 
            line=dict(color='darkgrey', width=2)
        ))
    
    # 9월 13일부터의 실제 데이터를 예측 데이터로 표시 - 파란색
    if not future_actual_data.empty:
        # 9월 12일 마지막 점부터 시작해서 연속적인 선을 만들기
        last_historical_date = historical_data['날짜'].iloc[-1] if not historical_data.empty else None
        last_historical_price = historical_data['가격'].iloc[-1] if not historical_data.empty else 0
        
        # 9월 12일 마지막 점을 포함한 연속적인 데이터 생성
        combined_dates = [last_historical_date]  # 9월 12일 마지막 점부터 시작
        combined_prices = [last_historical_price]
        
        first_future_date = future_actual_data['날짜'].iloc[0]
        first_future_price = future_actual_data['가격'].iloc[0]
        
        # 9월 13일, 14일 더미 데이터 생성 (9월 12일과 15일 사이를 부드럽게 연결)
        # 조건을 더 명확하게 수정: 9월 12일과 첫 번째 미래 데이터 사이에 빈 구간이 있는지 확인
        if first_future_date > pd.Timestamp('2025-09-12') + pd.Timedelta(days=1):
            # 9월 12일부터 첫 번째 미래 데이터까지의 가격을 선형 보간
            date_12 = last_historical_date
            date_future = first_future_date
            price_12 = last_historical_price
            price_future = first_future_price
            
            # 9월 12일과 첫 번째 미래 데이터 사이의 모든 날짜에 대해 더미 데이터 생성
            days_diff = (date_future - date_12).days
            price_diff = price_future - price_12
            
            for i in range(1, days_diff):
                dummy_date = date_12 + pd.Timedelta(days=i)
                dummy_price = price_12 + (price_diff * i / days_diff)
                combined_dates.append(dummy_date)
                combined_prices.append(dummy_price)
        
        # 실제 미래 데이터 추가
        combined_dates.extend(future_actual_data['날짜'].tolist())
        combined_prices.extend(future_actual_data['가격'].tolist())
        
        fig.add_trace(go.Scatter(
            x=combined_dates, 
            y=combined_prices, 
            mode='lines', 
            name='예측 데이터', 
            line=dict(color='royalblue', width=3)
        ))
    
    # 추가 예측 데이터 (9월 18일 이후) - 파란색
    if not predictions.empty:
        future_predictions = predictions[predictions['날짜'] > pd.Timestamp('2025-09-18')]
        if not future_predictions.empty:
            # 마지막 실제 데이터와 연결
            if not future_actual_data.empty:
                last_actual_point = future_actual_data.tail(1)
                connected_predictions = pd.concat([last_actual_point, future_predictions], ignore_index=True)
            else:
                connected_predictions = future_predictions
            
            fig.add_trace(go.Scatter(
                x=connected_predictions['날짜'], 
                y=connected_predictions['가격'], 
                mode='lines', 
                name='추가 예측', 
                line=dict(color='royalblue', width=3)
            ))
    # 예측 기간에 맞게 x축 범위 설정
    if st.session_state.predict_days > 0:
        # 기준일(9월 12일)부터 예측 기간만큼의 범위로 제한
        start_date = pd.Timestamp('2025-09-12')
        end_date = start_date + pd.Timedelta(days=st.session_state.predict_days)
        
        # 그래프의 x축 범위를 설정
        fig.update_layout(
            xaxis=dict(
                range=[start_date - pd.Timedelta(days=30), end_date],  # 30일 전부터 예측 기간까지
                showgrid=True
            )
        )
    
    fig.update_layout(title=dict(text=f'{item_name} ({unit}) 가격 추이 및 예측', x=0.5), yaxis_title=f'가격 (원/{unit})', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
    st.plotly_chart(fig, width='stretch')

    # 간단한 요약 지표
    if item_name == '쌀':
        # 쌀의 경우 9월 12일 기준으로 현재 가격과 예측 가격 계산
        cutoff_date = pd.Timestamp('2025-09-12')
        current_price = int(historical_data['가격'].iloc[-1])  # 9월 12일 가격
        
        # 예측 시작일 계산 (9월 12일 + predict_days)
        prediction_start_date = cutoff_date + pd.Timedelta(days=1)
        prediction_end_date = prediction_start_date + pd.Timedelta(days=predict_days-1)
        
        # 예측 데이터에서 해당 날짜의 가격 찾기
        if not predictions.empty:
            # 예측 데이터에서 해당 날짜 찾기
            pred_on_date = predictions[predictions['날짜'] == prediction_end_date]
            if not pred_on_date.empty:
                future_price = int(pred_on_date['가격'].iloc[0])
            else:
                # 정확한 날짜가 없으면 마지막 예측 가격 사용
                future_price = int(predictions['가격'].iloc[-1])
        else:
            future_price = current_price
    else:
        # 다른 품목들은 기존 방식 사용
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

# ==============================================================================
# 🛒 식자재 바로 구매 페이지 함수
# ==============================================================================
def purchase_page():
    # 상단 여백 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 뒤로가기 버튼 (아이콘만)
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("←", key="back_purchase", help="메인으로 돌아가기"):
            st.session_state.page = 'main'
            st.rerun()
    
    st.title("🛒 식자재 바로 구매하기")
    st.markdown("필요한 식자재를 즉시 구매하세요.")
    st.divider()
    
    # 구매 상품 목록
    st.subheader("📋 구매 가능한 상품")
    
    # 주요 품목별 구매 옵션
    items = ["쌀", "깐마늘(국산)", "양파"]
    item_icons = {"쌀": "🍚", "깐마늘(국산)": "🧄", "양파": "🧅"}
    item_units = {"쌀": "20kg", "깐마늘(국산)": "1kg", "양파": "10kg"}
    
    for item in items:
        with st.container(border=True):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.markdown(f"<h4>{item_icons[item]} {item}</h4>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**단위:** {item_units[item]}")
                st.markdown("**배송:** 당일 배송 가능")
            with col3:
                if st.button(f"구매하기", key=f"buy_{item}"):
                    st.success(f"{item} 구매 페이지로 이동합니다!")
    
    st.divider()
    st.info("💡 AI 구매 팁을 참고하여 최적의 구매 시점을 선택하세요!")

# ==============================================================================
# 📅 식자재 예약 구매 페이지 함수
# ==============================================================================
def reservation_page():
    # 상단 여백 추가
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 뒤로가기 버튼 (아이콘만)
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("←", key="back_reservation", help="메인으로 돌아가기"):
            st.session_state.page = 'main'
            st.rerun()
    
    st.title("📅 식자재 예약 구매하기")
    st.markdown("핵심 수익 모델: 미래 가격 예측을 통한 예약 구매로 비용 절약")
    st.divider()
    
    # 예약 구매 설명
    st.subheader("💰 예약 구매의 장점")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **💵 비용 절약**
        - AI 예측 기반 최적 가격에 구매
        - 시장 변동성 리스크 최소화
        """)
    
    with col2:
        st.markdown("""
        **📊 안정적 공급**
        - 미리 확정된 가격으로 예산 관리
        - 계절성 변동 대비
        """)
    
    with col3:
        st.markdown("""
        **🎯 전략적 구매**
        - 데이터 기반 의사결정
        - 경쟁 우위 확보
        """)
    
    st.divider()
    
    # 예약 구매 상품 선택
    st.subheader("📋 예약 구매 상품 선택")
    
    items = ["쌀", "깐마늘(국산)", "양파"]
    item_icons = {"쌀": "🍚", "깐마늘(국산)": "🧄", "양파": "🧅"}
    
    selected_item = st.selectbox("예약 구매할 상품을 선택하세요:", items)
    
    if selected_item:
        st.markdown(f"### {item_icons[selected_item]} {selected_item} 예약 구매")
        
        # 예약 기간 선택
        col1, col2 = st.columns(2)
        with col1:
            reservation_days = st.selectbox("예약 기간", [7, 14, 30], format_func=lambda x: f"{x}일 후")
        with col2:
            quantity = st.number_input("수량", min_value=1, max_value=100, value=1)
        
        # 예약 구매 버튼
        if st.button("📅 예약 구매 신청", type="primary", use_container_width=True):
            st.success(f"{selected_item} {quantity}개를 {reservation_days}일 후 예약 구매 신청이 완료되었습니다!")
            st.info("AI가 최적의 가격을 찾아 자동으로 구매를 진행합니다.")

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
elif st.session_state.page == 'purchase':
    purchase_page()
elif st.session_state.page == 'reservation':
    reservation_page()

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

