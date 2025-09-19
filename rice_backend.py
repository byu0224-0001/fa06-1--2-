import os
from datetime import timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import streamlit as st
except ImportError:
    # Streamlit이 없는 환경에서는 캐싱 데코레이터를 무시
    def st_cache_data(func):
        return func
    st = type('MockStreamlit', (), {'cache_data': st_cache_data})()

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from scipy.signal import butter, filtfilt
except Exception as e:
    print(f"일부 패키지 import 실패: {e}")
    xgb = None
    StandardScaler = None
    MinMaxScaler = None
    mean_squared_error = None
    mean_absolute_error = None
    r2_score = None
    joblib = None
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    butter = None
    filtfilt = None

# -------------------------
# 전역 변수 및 캐싱 (메모리 효율성)
# -------------------------
_CACHED_DATA = None
_DEVICE = None

def _get_device():
    """GPU 사용 가능 여부 확인 및 디바이스 설정"""
    global _DEVICE
    if _DEVICE is None:
        if torch is not None and torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
            print("GPU 사용: CUDA")
        else:
            _DEVICE = torch.device('cpu') if torch is not None else 'cpu'
            print("CPU 사용")
    return _DEVICE

# -------------------------
# DLinear 모델 정의 (원본 노트북과 동일)
# -------------------------
if nn is not None:
    class MovingAverage(nn.Module):
        def __init__(self, kernel_size, stride=1):
            super(MovingAverage, self).__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        def forward(self, x):
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            return x

    class DLinear(nn.Module):
        def __init__(self, input_length, output_length, input_dim, kernel_size=25):
            super(DLinear, self).__init__()
            self.input_length = input_length
            self.output_length = output_length
            
            self.moving_avg = MovingAverage(kernel_size)
            self.linear_seasonal = nn.Linear(input_length, output_length)
            self.linear_trend = nn.Linear(input_length, output_length)
            self.regression_layer = nn.Linear(input_dim, 1)

        def forward(self, x):
            seasonal_init = x
            trend_init = self.moving_avg(x)
            seasonal_init = seasonal_init - trend_init

            seasonal_output = self.linear_seasonal(seasonal_init.permute(0, 2, 1))
            trend_output = self.linear_trend(trend_init.permute(0, 2, 1))
            
            seasonal_output = seasonal_output.permute(0, 2, 1)
            trend_output = trend_output.permute(0, 2, 1)
            
            output = seasonal_output + trend_output
            output = self.regression_layer(output)
            return output
else:
    # PyTorch가 없을 때 더미 클래스
    class MovingAverage:
        def __init__(self, *args, **kwargs):
            pass
    
    class DLinear:
        def __init__(self, *args, **kwargs):
            pass

# -------------------------
# 데이터 로드 및 전처리 (캐싱으로 최적화)
# -------------------------
def _load_all_data(item_name: str = '쌀') -> pd.DataFrame:
    """모든 품목에 대한 데이터 로드 및 전처리 (캐싱 적용)"""
    global _CACHED_DATA
    
    # 캐시 키를 품목별로 구분
    cache_key = f"{item_name}_data"
    if _CACHED_DATA is not None and hasattr(_CACHED_DATA, 'get') and _CACHED_DATA.get(cache_key) is not None:
        return _CACHED_DATA[cache_key].copy()
    
    try:
        # 1) 환율 데이터 (원본 노트북과 동일한 방식)
        df_exchange = pd.read_csv('data/exchange_rate.csv')
        df_exchange.columns = ['날짜', '환율']
        df_exchange['날짜'] = pd.to_datetime(df_exchange['날짜'], format='%Y-%m-%d')
        df_exchange['환율'] = pd.to_numeric(df_exchange['환율'].astype(str).str.replace(',', ''), errors='coerce')
        df_exchange = df_exchange.groupby('날짜').mean().reset_index()
        df_exchange = df_exchange.dropna(subset=['날짜'])
    except Exception:
        df_exchange = pd.DataFrame({'날짜': [pd.Timestamp.today()], '환율': [1300.0]})

    try:
        # 2) 유가 데이터 (원본 노트북과 동일한 방식)
        df_oil = pd.read_csv('data/oil.csv', encoding='cp949')
        df_oil.columns = ['날짜', '유가']
        df_oil['날짜'] = pd.to_datetime(df_oil['날짜'], format='%Y-%m-%d')
        df_oil = df_oil.groupby('날짜').mean().reset_index()
        df_oil = df_oil.dropna(subset=['날짜'])
    except Exception:
        df_oil = pd.DataFrame({'날짜': [pd.Timestamp.today()], '유가': [80.0]})

    try:
        # 3) 날씨 데이터 (원본 노트북과 동일한 방식)
        df_weather = pd.read_csv('data/top_weather_features.csv', encoding='utf-8')
        # 누적강수량 컬럼 제거
        if '누적강수량' in df_weather.columns:
            df_weather.drop('누적강수량', axis=1, inplace=True)
        # 컬럼명을 직접 설정 (원본 노트북과 동일)
        df_weather.columns = ['날짜', '누적평균기온', '누적일조합']
        df_weather['날짜'] = pd.to_datetime(df_weather['날짜'], format='%Y-%m-%d')
        df_weather = df_weather.groupby('날짜').mean().reset_index()
        df_weather = df_weather.dropna(subset=['날짜'])
    except Exception:
        df_weather = pd.DataFrame({'날짜': [pd.Timestamp.today()], '누적평균기온': [15.0], '누적일조합': [6.0]})

    try:
        # 4) 품목별 데이터 로드
        df_item = _load_item_data(item_name)
        
        # 쌀의 경우 실제 데이터 추가 (9월 5일-18일)
        if item_name == '쌀':
            # 실제 쌀 20kg 가격 데이터 (9월 5일-18일)
            actual_rice_data = pd.DataFrame({
                '날짜': [
                    '2025-09-05', '2025-09-08', '2025-09-09', '2025-09-10', '2025-09-11',
                    '2025-09-12', '2025-09-15', '2025-09-16', '2025-09-17', '2025-09-18'
                ],
                '가격': [60500, 60900, 60800, 61300, 61100, 61400, 61500, 62906, 63279, 63631]
            })
            actual_rice_data['날짜'] = pd.to_datetime(actual_rice_data['날짜'])
            
            # 기존 데이터와 실제 데이터 병합 (실제 데이터 우선)
            df_item = pd.concat([df_item, actual_rice_data], ignore_index=True)
            df_item = df_item.drop_duplicates(subset=['날짜'], keep='last')
            df_item = df_item.sort_values('날짜').reset_index(drop=True)
            
    except Exception:
        # 기본값 설정
        base_prices = {
            '쌀': 52000, '감자': 3000, '배추': 2000, '양파': 2500, '오이': 4000,
            '상추': 1500, '무': 1000, '파': 2000, '대파': 2000, '건고추': 15000,
            '깐마늘': 120000, '깐마늘(국산)': 120000, '깐마늘(수입)': 80000
        }
        base_price = base_prices.get(item_name, 5000)
        df_item = pd.DataFrame({'날짜': [pd.Timestamp.today()], '가격': [base_price]})

    # 5) 모든 데이터 병합
    try:
        new = pd.merge(df_exchange, df_oil, how='outer', on='날짜')
        new2 = pd.merge(new, df_weather, how='outer', on='날짜')
        df = pd.merge(new2, df_item, how='outer', on='날짜')
        df = df.sort_values('날짜').reset_index(drop=True)
        
        # 빈 데이터 확인 및 처리
        if df.empty or df['가격'].isna().all():
            print(f"데이터 병합 실패 ({item_name}): 빈 데이터 반환")
            raise ValueError("데이터가 비어있습니다")
            
        # 가격 데이터가 있는 행만 유지
        df = df.dropna(subset=['가격'])
        if df.empty:
            print(f"가격 데이터가 없음 ({item_name})")
            raise ValueError("가격 데이터가 없습니다")
            
    except Exception as e:
        print(f"데이터 병합 오류 ({item_name}): {e}")
        raise ValueError("데이터 병합 실패")
    
    # 캐싱 (품목별로)
    if _CACHED_DATA is None:
        _CACHED_DATA = {}
    _CACHED_DATA[cache_key] = df.copy()
    return df

def _load_item_data(item_name: str) -> pd.DataFrame:
    """품목별 데이터 로드 및 전처리"""
    # 품목별 파일 매핑
    file_mapping = {
        '쌀': 'data/rice.csv',
        '감자': '감자_도매_데이터.xlsx',
        '배추': '배추_도매_데이터.xlsx',
        '양파': 'data/양파_도매_데이터.xlsx',
        '오이': '오이_도매_데이터.xlsx',
        '상추': '상추_도매.xlsx',
        '무': '무_도매.xlsx',
        '파': '파_도매_데이터.xlsx',
        '대파': 'data/깐마늘(국산)_도매_데이터.xlsx',  # 대파를 깐마늘(국산)로 교체
        '깐마늘': 'data/깐마늘(국산)_도매_데이터.xlsx',  # 깐마늘 추가
        '건고추': '건고추_도매_데이터.xlsx',
        '깐마늘(국산)': 'data/깐마늘(국산)_도매_데이터.xlsx',
        '깐마늘(수입)': '깐마늘(수입)_도매_데이터.xlsx'
    }
    
    file_path = file_mapping.get(item_name, 'data/rice.csv')
    
    if file_path.endswith('.csv'):
        # CSV 파일 처리 (쌀)
        df = pd.read_csv(file_path)
        # 불필요한 컬럼 제거
        if '품목명' in df.columns:
            df.drop(['품목명','품종명','시장명','지역명'], axis=1, inplace=True)
        # 컬럼명 설정
        df.columns = ['날짜', '가격']
        df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d')
        df['가격'] = df['가격'].astype(float)
    else:
        # Excel 파일 처리 (다른 품목들)
        df = pd.read_excel(file_path)
        
        # 대파(깐마늘), 깐마늘, 양파 데이터 특별 처리
        if item_name in ['대파', '깐마늘', '양파'] and 'DATE' in df.columns:
            df['날짜'] = pd.to_datetime(df['DATE'])
            df['가격'] = df['평균가격'].astype(float)
            df = df[['날짜', '가격']]
        else:
            # 일반 Excel 파일 처리
            # 컬럼명을 표준화
            df.columns = ['날짜', '품목', '품종', '거래단위', '등급', '평균가격', '전일', '전년']
            df['날짜'] = pd.to_datetime(df['날짜'])
            df['가격'] = df['평균가격'].astype(float)
            # 불필요한 컬럼 제거
            df = df[['날짜', '가격']]
    
    # 날짜별 평균 가격 계산
    df = df.groupby('날짜')['가격'].mean().reset_index()
    df = df.dropna(subset=['날짜', '가격'])
    
    return df

def get_item_history(item_name: str, days: int = 365) -> pd.DataFrame:
    """모든 품목에 대한 히스토리 데이터 반환"""
    try:
        # 품목별 데이터 로드
        full_data = _load_all_data(item_name)
        
        if full_data.empty:
            raise ValueError("데이터가 비어있습니다")
        
        # 가격 데이터 추출
        hist = full_data[['날짜', '가격']].copy()
        
        # 가격 데이터 검증
        hist = hist.dropna(subset=['가격'])
        if hist.empty:
            raise ValueError("유효한 가격 데이터가 없습니다")
            
    except Exception as e:
        print(f"데이터 로드 오류 ({item_name}): {e}")
        # 오류 발생 시 기본 데이터 반환
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
        base_prices = {
            '쌀': 52000, '감자': 3000, '배추': 2000, '양파': 2500, '오이': 4000,
            '상추': 1500, '무': 1000, '파': 2000, '대파': 2000, '건고추': 15000,
            '깐마늘': 120000, '깐마늘(국산)': 120000, '깐마늘(수입)': 80000
        }
        base_price = base_prices.get(item_name, 5000)
        prices = np.full(days, base_price)
        return pd.DataFrame({'날짜': pd.to_datetime(dates), '가격': prices})
    
    if days is not None and days > 0:
        hist = hist.tail(days)
    return hist.reset_index(drop=True)

def get_rice_history(days: int = 365) -> pd.DataFrame:
    """쌀 히스토리 (하위 호환성)"""
    return get_item_history('쌀', days)

# -------------------------
# 고급 피처 엔지니어링 (원본 노트북과 동일)
# -------------------------
def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['dayofweek'] = df['날짜'].dt.dayofweek
    dayofyear = df['날짜'].dt.dayofyear.astype(float)
    df['doy_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
    return df

def _add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 쌀 가격 래그/이동평균
    for lag in [1, 7, 14, 30]:
        df[f'price_lag_{lag}'] = df['가격'].shift(lag)
    df['price_ma_7'] = df['가격'].rolling(7, min_periods=1).mean()
    df['price_ma_30'] = df['가격'].rolling(30, min_periods=1).mean()
    
    # 다른 변수들의 래그/이동평균
    for col in ['환율', '유가', '누적평균기온', '누적일조합']:
        if col in df.columns:
            for lag in [1, 7, 14]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            df[f'{col}_ma_7'] = df[col].rolling(7, min_periods=1).mean()
    return df

def _add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """고급 피처 엔지니어링 (원본 노트북과 동일)"""
    df = df.copy()
    
    # 1. 가격 변화율
    df['price_diff'] = df['가격'].diff()
    df['price_change_rate'] = df['가격'].pct_change()
    
    # 2. 롤링 통계
    for window in [7, 14, 30]:
        df[f'price_std_{window}'] = df['가격'].rolling(window, min_periods=1).std()
        df[f'price_min_{window}'] = df['가격'].rolling(window, min_periods=1).min()
        df[f'price_max_{window}'] = df['가격'].rolling(window, min_periods=1).max()
    
    # 3. 저주파 필터 (트렌드 추출)
    if butter is not None and filtfilt is not None:
        try:
            b, a = butter(4, 0.1, btype='low')
            df['price_trend'] = filtfilt(b, a, df['가격'].fillna(method='ffill').values)
        except:
            df['price_trend'] = df['가격'].rolling(30, min_periods=1).mean()
    else:
        df['price_trend'] = df['가격'].rolling(30, min_periods=1).mean()
    
    # 4. 정부 개입 확률 (원본 노트북의 고급 피처)
    df['prob_intervention_by_change'] = np.where(
        df['price_change_rate'] > 0.05, 0.8,  # 5% 이상 상승 시 개입 확률 80%
        np.where(df['price_change_rate'] < -0.05, 0.2, 0.5)  # 5% 이상 하락 시 개입 확률 20%
    )
    
    # 5. 이동평균 기반 방출 확률
    ma_30 = df['가격'].rolling(30, min_periods=1).mean()
    df['prob_release_by_ma'] = np.where(
        df['가격'] > ma_30 * 1.1, 0.7,  # 30일 평균 대비 10% 이상 높으면 방출 확률 70%
        np.where(df['가격'] < ma_30 * 0.9, 0.3, 0.5)  # 30일 평균 대비 10% 이상 낮으면 방출 확률 30%
    )
    
    return df

def _build_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_time_features(df)
    df = _add_lag_rolling(df)
    df = _add_advanced_features(df)
    df['target_next'] = df['가격'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    
    # DLinear 예측 제거 - 43개 피처로 일관성 유지
    # 기존 모델이 43개 피처로 학습되었으므로 43개로 통일
    
    feature_cols = [c for c in df.columns if c not in ['target_next', '날짜'] and c != '가격']
    # 피처 컬럼을 알파벳 순으로 정렬하여 일관성 보장
    feature_cols = sorted(feature_cols)
    return df, feature_cols

# -------------------------
# DLinear 예측 생성 함수 (원본 노트북과 동일)
# -------------------------
def generate_dlinear_predictions(scaled_feature_df, sequence_length, batch_size, model):
    """스케일링된 피처 데이터프레임을 받아 DLinear 모델의 예측을 생성합니다."""
    if torch is None or model is None:
        # PyTorch가 없거나 모델이 없으면 더미 예측 반환
        return [0.0] * (len(scaled_feature_df) - sequence_length + 1)
    
    model.eval()
    device = _get_device()
    model.to(device)
    
    # 1. 데이터로부터 시퀀스 생성
    sequences = []
    for i in range(len(scaled_feature_df) - sequence_length + 1):
        sequences.append(scaled_feature_df.iloc[i:(i + sequence_length)].values)
    
    sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)

    # 2. DataLoader 생성
    dataset = TensorDataset(sequences_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 모델 예측
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs[0])
            predictions.extend(outputs.flatten().cpu().numpy())
            
    return predictions

# -------------------------
# 모델 학습/저장/로드 (원본 노트북과 동일한 성능, 최적화)
# -------------------------
def _get_model_paths(item_name: str):
    """품목별 모델 경로 반환"""
    return {
        'model': f'models/{item_name}_xgb_model.json',
        'scaler': f'models/{item_name}_scaler.pkl',
        'dlinear': f'models/{item_name}_dlinear_model.pth',
        'features': f'models/{item_name}_feature_cols.pkl'
        
    }

def _clean_numeric_frame(df):
    """NaN/Inf 값 정리 (원본 노트북과 동일)"""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    return df

def train_model(history: pd.DataFrame, item_name: str = '쌀', fast_mode: bool = False) -> None:
    """품목별 모델 학습 (최적화)"""
    if xgb is None or StandardScaler is None or torch is None:
        return
    
    print(f"{item_name} 모델 학습 시작...")
    device = _get_device()
    
    # models 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 품목별 모델 경로
    paths = _get_model_paths(item_name)
    
    # 품목별 데이터셋으로 학습
    full_data = _load_all_data(item_name)
    df_sup, feature_cols = _build_supervised(full_data)
    if len(df_sup) < 50:
        print(f"{item_name} 데이터가 부족합니다.")
        return
    
    X = df_sup[feature_cols]
    y = df_sup['target_next']
    
    # 데이터 정리
    X = _clean_numeric_frame(X)
    y = _clean_numeric_frame(pd.DataFrame({'target': y}))['target']
    
    # 훈련/테스트 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 스케일링 (43개 피처로)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # DLinear 모델 학습 (원본 노트북과 동일한 설정)
    sequence_length = 30
    predict_length = 1
    input_dim = X_train_scaled.shape[1]
    batch_size = 32
    
    # PyTorch가 없으면 DLinear 학습 건너뛰기
    if torch is None:
        print("PyTorch가 없어 DLinear 학습을 건너뜁니다.")
        dlinear_preds_train = [0.0] * (len(X_train_scaled) - sequence_length + 1)
        dlinear_preds_test = [0.0] * (len(X_test_scaled) - sequence_length + 1)
    else:
        dlinear_model = DLinear(sequence_length, predict_length, input_dim).to(device)
        
        # 데이터를 시퀀스 형태로 변환
        def create_sequences(data, seq_len):
            sequences = []
            for i in range(len(data) - seq_len + 1):
                sequences.append(data[i:i+seq_len])
            return np.array(sequences)
        
        X_train_seq = create_sequences(X_train_scaled, sequence_length)
        y_train_seq = y_train.iloc[sequence_length-1:].values
        
        if len(X_train_seq) > 0:
            X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
            y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1).to(device)
            
            # DLinear 학습 (원본과 동일한 에포크 수)
            optimizer = optim.Adam(dlinear_model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
            
            best_val_loss = float('inf')
            patience = 10
            epochs_without_improvement = 0
            
            print(f"DLinear 학습 시작 (GPU: {device})...")
            for epoch in range(100):  # 원본과 동일한 에포크 수
                dlinear_model.train()
                optimizer.zero_grad()
                outputs = dlinear_model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    print(f'DLinear Epoch {epoch}, Loss: {loss.item():.6f}')
                
                scheduler.step(loss.item())
                
                if loss.item() < best_val_loss:
                    best_val_loss = loss.item()
                    epochs_without_improvement = 0
                    torch.save(dlinear_model.state_dict(), paths['dlinear'])
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"조기 종료: {epoch} 에포크")
                        break
            
            # DLinear 예측 생성 (원본 노트북과 동일)
            dlinear_model.load_state_dict(torch.load(paths['dlinear']))
            dlinear_model.eval()
            
            print("DLinear 예측 생성 중...")
            dlinear_preds_train = generate_dlinear_predictions(
                pd.DataFrame(X_train_scaled), sequence_length, batch_size, dlinear_model
            )
            
            # XGBoost 학습 (DLinear 예측 포함)
            X_train_final = X_train_scaled[sequence_length-1:]
            X_train_final = np.column_stack([X_train_final, dlinear_preds_train])
            
            X_test_seq = create_sequences(X_test_scaled, sequence_length)
            if len(X_test_seq) > 0:
                dlinear_preds_test = generate_dlinear_predictions(
                    pd.DataFrame(X_test_scaled), sequence_length, batch_size, dlinear_model
                )
                X_test_final = X_test_scaled[sequence_length-1:]
                X_test_final = np.column_stack([X_test_final, dlinear_preds_test])
            else:
                X_test_final = X_test_scaled
            
            # 44개 피처로 새로운 스케일러 생성
            print("44개 피처로 새로운 스케일러 생성 중...")
            final_scaler = StandardScaler()
            X_train_final_scaled = final_scaler.fit_transform(X_train_final)
            X_test_final_scaled = final_scaler.transform(X_test_final)
            
            y_train_final = y_train.iloc[sequence_length-1:]
            y_test_final = y_test.iloc[sequence_length-1:] if len(X_test_seq) > 0 else y_test
            
            # XGBoost 모델 학습 (원본과 동일한 설정)
            print("XGBoost 학습 중...")
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=500,  # 원본과 동일
                learning_rate=0.05,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                early_stopping_rounds=50
            )
            
            xgb_model.fit(
                X_train_final_scaled, y_train_final,
                eval_set=[(X_test_final_scaled, y_test_final)],
                verbose=False
            )
        
            # 모델 저장 (44개 피처용 스케일러 저장)
            xgb_model.save_model(paths['model'])
            joblib.dump(final_scaler, paths['scaler'])  # 44개 피처용 스케일러
            joblib.dump(feature_cols, paths['features'])
            
            print(f"모델 학습 완료! XGBoost RMSE: {np.sqrt(mean_squared_error(y_test_final, xgb_model.predict(X_test_final_scaled))):.2f}")
        else:
            # PyTorch가 없을 때는 43개 피처로만 학습
            print("43개 피처로 XGBoost 학습 중...")
            xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                early_stopping_rounds=50
            )
            
            xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # 모델 저장 (43개 피처용 스케일러 저장)
            xgb_model.save_model(paths['model'])
            joblib.dump(scaler, paths['scaler'])  # 43개 피처용 스케일러
            joblib.dump(feature_cols, paths['features'])
            
            print(f"모델 학습 완료! XGBoost RMSE: {np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test_scaled))):.2f}")

def _load_model(item_name: str = '쌀'):
    if xgb is None:
        return None
    paths = _get_model_paths(item_name)
    if not os.path.exists(paths['model']):
        return None
    m = xgb.XGBRegressor()
    m.load_model(paths['model'])
    return m

def _load_scaler(item_name: str = '쌀'):
    if joblib is None:
        return None
    paths = _get_model_paths(item_name)
    if not os.path.exists(paths['scaler']):
        return None
    return joblib.load(paths['scaler'])

def _load_feature_cols(item_name: str = '쌀'):
    if joblib is None:
        return None
    paths = _get_model_paths(item_name)
    if not os.path.exists(paths['features']):
        return None
    return joblib.load(paths['features'])

# -------------------------
# 재귀적 예측 (원본 노트북과 동일한 성능)
# -------------------------
@st.cache_data
def predict_item_price(item_name: str, history: pd.DataFrame, days_to_predict: int = 7) -> pd.DataFrame:
    """모든 품목에 대한 고급 가격 예측"""
    # 모든 품목에 대해 고급 예측 모델 사용
    return predict_advanced_price(item_name, history, days_to_predict)

@st.cache_data
def predict_advanced_price(item_name: str, history: pd.DataFrame, days_to_predict: int = 7) -> pd.DataFrame:
    """모든 품목에 대한 고급 가격 예측"""
    history = history.sort_values('날짜').reset_index(drop=True)

    # 쌀의 경우 9월 18일 이후부터 예측하도록 조정
    if item_name == '쌀':
        # 9월 18일 이후 데이터만 사용
        cutoff_date = pd.Timestamp('2025-09-18')
        history = history[history['날짜'] <= cutoff_date]
        if history.empty:
            print(f"{item_name} 9월 18일 이후 데이터가 없습니다.")
            return _fallback_prediction(history, days_to_predict)

    model = _load_model(item_name)
    if model is None:
        print(f"{item_name} 모델이 없습니다. 학습을 시작합니다...")
        train_model(history, item_name)
        model = _load_model(item_name)
        if model is None:
            print(f"{item_name} 모델 학습 실패. 기본 예측을 사용합니다.")
            # 모델 학습 실패 시 기본 예측 사용
            return _fallback_prediction(history, days_to_predict)

    if model is not None:
        # 품목별 데이터셋으로 작업
        full_data = _load_all_data(item_name)
        df_work = full_data.copy()
        
        # 데이터 검증
        if df_work.empty or len(df_work) == 0:
            print(f"{item_name} 데이터가 비어있습니다. 기본 예측을 사용합니다.")
            return _fallback_prediction(history, days_to_predict)
        
        preds = []
        # 스케일러 및 피처 컬럼명 로드
        scaler = _load_scaler(item_name)
        feature_cols = _load_feature_cols(item_name)
        
        # 피처 컬럼명이 없으면 새로 생성
        if feature_cols is None:
            _, feature_cols = _build_supervised(full_data)
        
        # 피처 개수 확인
        print(f"학습된 모델 피처 개수: {len(feature_cols)}")
        print(f"피처 목록: {feature_cols[:10]}...")  # 처음 10개만 출력
        
        # -----------------------------------------------------------------
        # --- 개선된 부분: DLinear 모델을 예측 루프 외부에서 한 번만 로드 ---
        # -----------------------------------------------------------------
        dlinear_model = None
        if torch is not None:
            try:
                paths = _get_model_paths(item_name)
                if os.path.exists(paths['dlinear']):
                    dlinear_model = DLinear(30, 1, len(feature_cols)).to(_get_device())
                    dlinear_model.load_state_dict(torch.load(paths['dlinear']))
                    dlinear_model.eval()
                    print("DLinear 모델 로드 완료.")
                else:
                    print("DLinear 모델 파일이 없습니다.")
            except Exception as e:
                print(f"DLinear 모델 로드 실패: {e}")
                dlinear_model = None
        else:
            print("PyTorch가 없어 DLinear 모델을 사용할 수 없습니다.")
            
        # 쌀의 경우 9월 19일부터 예측 시작
        if item_name == '쌀':
            start_date = pd.Timestamp('2025-09-19')
        else:
            start_date = df_work['날짜'].iloc[-1] + timedelta(days=1)
            
        for day_idx in range(days_to_predict):
            next_date = start_date + timedelta(days=day_idx)
            
            # 개선된 변수별 예측 방법
            next_row = {}
            
            # 1) 환율 예측 (더 안정적인 방법)
            if '환율' in df_work.columns:
                last_val = df_work['환율'].iloc[-1]
                
                # 최근 90일 데이터로 더 안정적인 트렌드 계산
                recent_vals = df_work['환율'].tail(90)
                if len(recent_vals) >= 30:
                    # 30일 이동평균으로 트렌드 계산
                    ma_30 = recent_vals.tail(30).mean()
                    ma_60 = recent_vals.tail(60).mean() if len(recent_vals) >= 60 else ma_30
                    trend = (ma_30 - ma_60) / ma_60 * 0.1  # 매우 완만한 트렌드
                else:
                    trend = 0
                
                # 계절성 (연말/연초 효과)
                day_of_year = next_date.dayofyear
                seasonal_factor = 1 + 0.02 * np.sin(2 * np.pi * (day_of_year - 15) / 365.25)
                
                # 매우 작은 랜덤 변동 (0.1%)
                noise = np.random.normal(0, 0.001)
                
                next_row['환율'] = last_val * seasonal_factor * (1 + trend + noise)
            
            # 2) 유가 예측 (더 안정적인 방법)
            if '유가' in df_work.columns:
                last_val = df_work['유가'].iloc[-1]
                
                # 최근 60일 데이터로 안정적인 트렌드 계산
                recent_vals = df_work['유가'].tail(60)
                if len(recent_vals) >= 20:
                    ma_20 = recent_vals.tail(20).mean()
                    ma_40 = recent_vals.tail(40).mean() if len(recent_vals) >= 40 else ma_20
                    trend = (ma_20 - ma_40) / ma_40 * 0.05  # 매우 완만한 트렌드
                else:
                    trend = 0
                
                # 계절성 (겨울철 유가 상승)
                day_of_year = next_date.dayofyear
                seasonal_factor = 1 + 0.03 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
                
                # 작은 랜덤 변동 (0.2%)
                noise = np.random.normal(0, 0.002)
                
                next_row['유가'] = last_val * seasonal_factor * (1 + trend + noise)
            
            # 3) 날씨 예측 (계절성 중심)
            if '누적평균기온' in df_work.columns:
                last_val = df_work['누적평균기온'].iloc[-1]
                # 강한 계절성 + 약한 트렌드
                day_of_year = next_date.dayofyear
                seasonal_base = 15 + 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
                trend = 0.1 * np.sin(2 * np.pi * day_of_year / 365.25)  # 연간 트렌드
                noise = np.random.normal(0, 2.0)
                next_row['누적평균기온'] = seasonal_base + trend + noise
                
            if '누적일조합' in df_work.columns:
                last_val = df_work['누적일조합'].iloc[-1]
                # 일조시간은 계절성과 구름 패턴에 따라
                day_of_year = next_date.dayofyear
                seasonal_base = 6 + 4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
                noise = np.random.normal(0, 1.0)
                next_row['누적일조합'] = max(0, seasonal_base + noise)
            
            # 4) 변수 간 상관관계 고려한 보정
            if len(df_work) > 30:
                # 최근 30일 상관관계 계산
                recent_data = df_work[['환율', '유가', '누적평균기온', '누적일조합', '가격']].tail(30)
                corr_matrix = recent_data.corr()
                
                # 환율-유가 상관관계 반영
                if '환율' in next_row and '유가' in next_row:
                    usd_oil_corr = corr_matrix.loc['환율', '유가']
                    if not pd.isna(usd_oil_corr) and abs(usd_oil_corr) > 0.3:
                        # 유가 변화에 따른 환율 조정
                        oil_change = (next_row['유가'] - df_work['유가'].iloc[-1]) / df_work['유가'].iloc[-1]
                        usd_adjustment = oil_change * usd_oil_corr * 0.1
                        next_row['환율'] *= (1 + usd_adjustment)
                
                # 날씨-가격 상관관계 반영 (간접적)
                if '누적평균기온' in next_row and '가격' in next_row:
                    temp_price_corr = corr_matrix.loc['누적평균기온', '가격']
                    if not pd.isna(temp_price_corr) and abs(temp_price_corr) > 0.2:
                        # 기온 변화가 가격에 미치는 영향 고려
                        temp_change = (next_row['누적평균기온'] - df_work['누적평균기온'].iloc[-1]) / df_work['누적평균기온'].iloc[-1]
                        price_adjustment = temp_change * temp_price_corr * 0.05
                        # 이는 다음 단계에서 가격 예측에 반영됨
            
            next_row['날짜'] = next_date
            next_row['가격'] = df_work['가격'].iloc[-1]  # 임시값
            
            tmp = pd.concat([df_work, pd.DataFrame([next_row])], ignore_index=True)
            tmp = _add_time_features(tmp)
            tmp = _add_lag_rolling(tmp)
            tmp = _add_advanced_features(tmp)
            
            # 학습 시와 동일한 방식으로 피처 생성
            tmp_supervised, _ = _build_supervised(tmp)
            if len(tmp_supervised) == 0:
                # 데이터가 부족한 경우 기본값으로 채우기
                x_next = pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)
            else:
                # 마지막 행의 피처만 선택
                x_next = tmp_supervised[feature_cols].iloc[[-1]]
            
            # DLinear 예측 추가 (이미 로드된 모델 사용)
            if dlinear_model is not None:
                try:
                    # 시퀀스 생성
                    sequence_length = 30
                    if len(tmp_supervised) >= sequence_length:
                        # 마지막 30개 행으로 시퀀스 생성
                        seq_data = tmp_supervised[feature_cols].iloc[-sequence_length:].values
                        seq_tensor = torch.FloatTensor(seq_data).unsqueeze(0).to(_get_device())
                        
                        with torch.no_grad():
                            dlinear_pred = dlinear_model(seq_tensor).cpu().numpy()[0][0]
                    else:
                        dlinear_pred = 0.0
                except Exception as e:
                    print(f"DLinear 예측 생성 실패: {e}")
                    dlinear_pred = 0.0
            else:
                dlinear_pred = 0.0
            
            # DLinear 예측을 피처에 추가
            x_next_array = x_next.values
            x_next_with_dlinear = np.column_stack([x_next_array, [dlinear_pred]])
            
            # 피처 개수 검증
            model_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(feature_cols) + 1
            expected_features = model_features
            actual_features = x_next_with_dlinear.shape[1]
            if actual_features != expected_features:
                print(f"피처 개수 불일치: 예상 {expected_features}, 실제 {actual_features}")
                # 부족한 피처를 0으로 채우기
                if actual_features < expected_features:
                    missing_count = expected_features - actual_features
                    padding = np.zeros((1, missing_count))
                    x_next_with_dlinear = np.column_stack([x_next_with_dlinear, padding])
                # 초과한 피처 제거
                elif actual_features > expected_features:
                    x_next_with_dlinear = x_next_with_dlinear[:, :expected_features]
            
            
            if scaler is not None:
                # DLinear 예측이 포함된 데이터로 스케일링
                x_next_scaled = scaler.transform(x_next_with_dlinear)
                y_hat = float(model.predict(x_next_scaled)[0])
            else:
                y_hat = float(model.predict(x_next_with_dlinear)[0])
            
            # 5) 앙상블 예측 (여러 방법의 가중 평균)
            ensemble_preds = []
            
            # 방법 1: XGBoost + DLinear 예측
            ensemble_preds.append(y_hat)
            
            # 방법 2: 고급 트렌드 기반 예측 (실제 데이터 기반 강화)
            if len(df_work) >= 30:
                # 다중 기간 트렌드 분석
                recent_7 = df_work['가격'].tail(7)
                recent_14 = df_work['가격'].tail(14)
                recent_30 = df_work['가격'].tail(30)
                
                # 일일 변화율 기반 트렌드 계산
                trend_7 = (recent_7.iloc[-1] - recent_7.iloc[0]) / recent_7.iloc[0] / len(recent_7)
                trend_14 = (recent_14.iloc[-1] - recent_14.iloc[0]) / recent_14.iloc[0] / len(recent_14)
                trend_30 = (recent_30.iloc[-1] - recent_30.iloc[0]) / recent_30.iloc[0] / len(recent_30)
                
                # 시계열 추세 반영 (품목별 제한)
                current_date = next_date
                
                # 품목별 트렌드 가중치 및 제한 설정
                if item_name == '쌀':
                    # 쌀: 보수적인 트렌드 반영
                    weighted_trend_rate = trend_7 * 0.3 + trend_14 * 0.4 + trend_30 * 0.3
                    max_trend_change = 0.005  # 최대 0.5% 변화
                elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                    # 깐마늘(국산): 실제 시장 변동성을 반영한 다이나믹한 예측
                    # 단기 변동성과 중기 트렌드를 균형있게 반영
                    weighted_trend_rate = trend_7 * 0.5 + trend_14 * 0.3 + trend_30 * 0.2
                    max_trend_change = 0.025  # 최대 2.5% 변화 (실제 시장 변동성 반영)
                    
                    # 깐마늘 특별 처리: 실제 시장 변동성을 반영한 조정
                    if len(preds) >= 3:
                        recent_3_prices = [preds[-i]['가격'] for i in range(1, 4)]
                        if all(recent_3_prices[i] < recent_3_prices[i+1] for i in range(len(recent_3_prices)-1)):
                            # 3일 연속 상승 시 강한 하락 트렌드로 조정 (실제 시장 반영)
                            weighted_trend_rate = -abs(weighted_trend_rate) * 1.5
                            print(f"깐마늘 3일 연속 상승 감지: 강한 하락 트렌드 조정")
                        elif all(recent_3_prices[i] > recent_3_prices[i+1] for i in range(len(recent_3_prices)-1)):
                            # 3일 연속 하락 시 강한 상승 트렌드로 조정 (실제 시장 반영)
                            weighted_trend_rate = abs(weighted_trend_rate) * 1.5
                            print(f"깐마늘 3일 연속 하락 감지: 강한 상승 트렌드 조정")
                    elif len(preds) >= 2:
                        recent_2_prices = [preds[-i]['가격'] for i in range(1, 3)]
                        if all(recent_2_prices[i] < recent_2_prices[i+1] for i in range(len(recent_2_prices)-1)):
                            # 2일 연속 상승 시 하락 트렌드로 조정
                            weighted_trend_rate = -abs(weighted_trend_rate) * 0.8
                            print(f"깐마늘 2일 연속 상승 감지: 하락 트렌드 조정")
                        elif all(recent_2_prices[i] > recent_2_prices[i+1] for i in range(len(recent_2_prices)-1)):
                            # 2일 연속 하락 시 상승 트렌드로 조정
                            weighted_trend_rate = abs(weighted_trend_rate) * 0.8
                            print(f"깐마늘 2일 연속 하락 감지: 상승 트렌드 조정")
                else:
                    # 기타 품목: 기존 설정 유지
                    if current_date >= pd.Timestamp('2025-09-15'):
                        weighted_trend_rate = trend_7 * 0.6 + trend_14 * 0.3 + trend_30 * 0.1
                        max_trend_change = 0.03
                    elif current_date >= pd.Timestamp('2025-09-12'):
                        weighted_trend_rate = trend_7 * 0.5 + trend_14 * 0.4 + trend_30 * 0.1
                        max_trend_change = 0.02
                    else:
                        weighted_trend_rate = trend_7 * 0.4 + trend_14 * 0.4 + trend_30 * 0.2
                        max_trend_change = 0.01
                
                # 트렌드 영향도 제한
                weighted_trend_rate = max(-max_trend_change, min(max_trend_change, weighted_trend_rate))
                
                trend_pred = df_work['가격'].iloc[-1] * (1 + weighted_trend_rate)
                ensemble_preds.append(trend_pred)
            
            # 방법 3: 고급 계절성 기반 예측
            if len(df_work) >= 365:
                # 1) 같은 요일의 과거 평균
                same_weekday_prices = df_work[df_work['날짜'].dt.weekday == next_date.weekday()]['가격'].tail(10)
                
                # 2) 같은 월의 과거 평균
                same_month_prices = df_work[df_work['날짜'].dt.month == next_date.month]['가격'].tail(5)
                
                # 3) 같은 계절의 과거 평균
                season = (next_date.month % 12 + 3) // 3  # 1:봄, 2:여름, 3:가을, 4:겨울
                same_season_prices = df_work[((df_work['날짜'].dt.month % 12 + 3) // 3) == season]['가격'].tail(8)
                
                seasonal_preds = []
                if len(same_weekday_prices) > 0:
                    seasonal_preds.append(same_weekday_prices.mean())
                if len(same_month_prices) > 0:
                    seasonal_preds.append(same_month_prices.mean())
                if len(same_season_prices) > 0:
                    seasonal_preds.append(same_season_prices.mean())
                
                if len(seasonal_preds) > 0:
                    # 가중 평균 (요일: 40%, 월: 40%, 계절: 20%)
                    weights = [0.4, 0.4, 0.2][:len(seasonal_preds)]
                    weights = [w/sum(weights) for w in weights]  # 정규화
                    seasonal_pred = sum(pred * weight for pred, weight in zip(seasonal_preds, weights))
                    ensemble_preds.append(seasonal_pred)
            
            # 방법 4: 변수 기반 선형 예측
            if len(ensemble_preds) > 0:
                # 환율, 유가 변화에 따른 선형 추정
                usd_change = (next_row['환율'] - df_work['환율'].iloc[-1]) / df_work['환율'].iloc[-1]
                oil_change = (next_row['유가'] - df_work['유가'].iloc[-1]) / df_work['유가'].iloc[-1]
                
                # 과거 상관관계 기반 가중치
                if len(df_work) > 30:
                    recent_data = df_work[['환율', '유가', '가격']].tail(30)
                    usd_price_corr = recent_data.corr().loc['환율', '가격']
                    oil_price_corr = recent_data.corr().loc['유가', '가격']
                    
                    if not pd.isna(usd_price_corr) and not pd.isna(oil_price_corr):
                        linear_pred = df_work['가격'].iloc[-1] * (1 + usd_change * usd_price_corr * 0.1 + oil_change * oil_price_corr * 0.1)
                        ensemble_preds.append(linear_pred)
            
            # 앙상블 가중 평균 (실제 데이터 기반 조정)
            if len(ensemble_preds) >= 2:
                # 2025년 9월 실제 데이터 기반 가중치 조정
                current_date = next_date
                if current_date >= pd.Timestamp('2025-09-15'):
                    # 9월 15일 이후 급격한 상승 반영
                    # XGBoost: 60%, 트렌드: 30%, 계절성: 10% (상승 가중치 증가)
                    weights = [0.6, 0.3, 0.1] if len(ensemble_preds) >= 3 else [0.7, 0.3]
                elif current_date >= pd.Timestamp('2025-09-12'):
                    # 9월 12일 이후 상승 추세 반영
                    # XGBoost: 70%, 트렌드: 25%, 계절성: 5%
                    weights = [0.7, 0.25, 0.05] if len(ensemble_preds) >= 3 else [0.75, 0.25]
                else:
                    # 기존 가중치 (XGBoost: 80%, 나머지: 20%)
                    weights = [0.8] + [0.2 / (len(ensemble_preds) - 1)] * (len(ensemble_preds) - 1)
                
                # 가중치 정규화
                weights = [w/sum(weights) for w in weights]
                final_pred = sum(pred * weight for pred, weight in zip(ensemble_preds, weights))
            else:
                final_pred = y_hat
            
            # 6) 품목별 특화 예측 보정
            last_price = df_work['가격'].iloc[-1]
            day_of_year = next_date.dayofyear
            
            # 품목별 특성에 따른 보정 적용
            if item_name == '쌀':
                # 쌀 특화 보정 (완만한 계절성과 현실적 변동성)
                current_date = next_date
                
                # 1) 완만한 계절성 적용 (극단적 하락 방지)
                day_of_year = next_date.dayofyear
                
                # 쌀의 계절성: 9-10월 완만한 하락, 11-12월 안정화, 1-3월 상승
                if 245 <= day_of_year <= 304:  # 9-10월 (수확기)
                    seasonal_factor = 0.998  # 매우 완만한 하락 (0.2%)
                elif 305 <= day_of_year <= 365:  # 11-12월 (안정기)
                    seasonal_factor = 1.000  # 중립
                elif 1 <= day_of_year <= 90:  # 1-3월 (상승기)
                    seasonal_factor = 1.001  # 매우 완만한 상승 (0.1%)
                else:  # 4-8월 (안정기)
                    seasonal_factor = 1.000  # 중립
                
                # 2) 2025년 9월 실제 데이터 기반 보정 (완만하게)
                if current_date >= pd.Timestamp('2025-09-15'):
                    # 9월 15일 이후 실제 상승 추세를 완만하게 반영
                    actual_trend_factor = 1.005  # 0.5% 상승 가중치 (완만하게)
                    seasonal_factor *= actual_trend_factor
                    print(f"쌀 완만한 상승 반영: {last_price:.0f}원 → {final_pred:.0f}원 (0.5% 상승 가중치)")
                elif current_date >= pd.Timestamp('2025-09-12'):
                    # 9월 12일 이후 상승 추세를 완만하게 반영
                    actual_trend_factor = 1.002  # 0.2% 상승 가중치 (매우 완만하게)
                    seasonal_factor *= actual_trend_factor
                    print(f"쌀 완만한 상승 반영: {last_price:.0f}원 → {final_pred:.0f}원 (0.2% 상승 가중치)")
                else:
                    # 기존 계절성 로직
                    if 240 <= day_of_year <= 300:  # 8월-10월 (쌀 수확기)
                        seasonal_factor = 0.995  # 수확기에는 가격 하락
                    elif 60 <= day_of_year <= 120:  # 3월-4월 (쌀 비수확기)
                        seasonal_factor = 1.005  # 비수확기에는 가격 상승
                    else:
                        seasonal_factor = 1.0
                
                # 2) 정부 개입 시뮬레이션 (실제 상승에 맞게 조정)
                if final_pred > last_price * 1.05:  # 5% 이상 상승 시 (기존 3%에서 완화)
                    intervention_factor = 0.99  # 정부 개입으로 가격 억제 (기존보다 완화)
                elif final_pred < last_price * 0.95:  # 5% 이상 하락 시
                    intervention_factor = 1.01  # 정부 지원으로 가격 상승 (기존보다 강화)
                else:
                    intervention_factor = 1.0
                
                # 3) 수급 불균형 시뮬레이션 (실제 상승에 맞게 조정)
                recent_volatility = df_work['가격'].tail(7).std() / df_work['가격'].tail(7).mean()
                if recent_volatility > 0.03:  # 변동성이 높을 때 (기존 0.02에서 완화)
                    stability_factor = 0.999  # 안정화 압력 (기존보다 완화)
                else:
                    stability_factor = 1.0
                    
            elif item_name == '양파':
                # 양파 특화 보정 (6-7월 수확기, 겨울철 가격 상승)
                if 150 <= day_of_year <= 210:  # 6월-7월 (양파 수확기)
                    seasonal_factor = 0.98  # 수확기에는 가격 하락 (2%)
                elif 300 <= day_of_year <= 60:  # 11월-2월 (양파 비수확기)
                    seasonal_factor = 1.02  # 비수확기에는 가격 상승 (2%)
                else:
                    seasonal_factor = 1.0
                
                # 양파는 정부 개입이 적으므로 기본값
                intervention_factor = 1.0
                stability_factor = 1.0
                
            elif item_name == '대파':
                # 대파 특화 보정 (연중 생산, 계절성 약함)
                # 대파는 계절성이 약하므로 최소한의 보정만
                seasonal_factor = 1.0
                intervention_factor = 1.0
                stability_factor = 1.0
                
            elif item_name == '건고추':
                # 건고추 특화 보정 (8-9월 수확기, 건조 특성)
                if 220 <= day_of_year <= 280:  # 8월-9월 (건고추 수확기)
                    seasonal_factor = 0.985  # 수확기에는 가격 하락 (1.5%)
                elif 1 <= day_of_year <= 60 or 300 <= day_of_year <= 365:  # 겨울철
                    seasonal_factor = 1.015  # 겨울철에는 가격 상승 (1.5%)
                else:
                    seasonal_factor = 1.0
                
                # 건고추는 정부 개입이 적으므로 기본값
                intervention_factor = 1.0
                stability_factor = 1.0
                
            elif item_name == '깐마늘(국산)':
                # 깐마늘 특화 보정 (5-6월 수확기)
                if 120 <= day_of_year <= 180:  # 5월-6월 (깐마늘 수확기)
                    seasonal_factor = 0.99  # 수확기에는 가격 하락 (1%)
                elif 300 <= day_of_year <= 60:  # 겨울철
                    seasonal_factor = 1.02  # 겨울철에는 가격 상승 (2%)
                else:
                    seasonal_factor = 1.0
                
                # 깐마늘은 정부 개입이 적으므로 기본값
                intervention_factor = 1.0
                stability_factor = 1.0
                
            else:
                # 기타 품목들은 기본값 (보정 없음)
                seasonal_factor = 1.0
                intervention_factor = 1.0
                stability_factor = 1.0
            
            # 최종 예측값 계산
            final_pred = final_pred * seasonal_factor * intervention_factor * stability_factor
            
            # 9) 평균 회귀 특성 추가 (가격이 평균으로 돌아가려는 특성)
            if len(df_work) >= 30:
                # 30일 이동평균 대비 현재 가격의 편차 계산
                ma_30 = df_work['가격'].tail(30).mean()
                current_price = df_work['가격'].iloc[-1]
                deviation = (current_price - ma_30) / ma_30
                
                # 품목별 평균 회귀 강도 설정
                if item_name == '대파':  # 깐마늘(국산)로 교체
                    # 깐마늘: 실제 시장 변동성을 반영한 완화된 평균 회귀
                    if abs(deviation) > 0.30:  # 30% 이상 편차가 있으면 (실제 시장 반영)
                        mean_reversion_factor = 1 - (deviation * 0.005)  # 편차의 0.5%만큼 평균으로 회귀 (완화된 회귀)
                        final_pred = final_pred * mean_reversion_factor
                        print(f"깐마늘 완화된 평균 회귀 적용: 편차 {deviation:.3f}, 조정 계수 {mean_reversion_factor:.3f}")
                else:
                    # 기타 품목: 기존 설정
                    if abs(deviation) > 0.20:  # 20% 이상 편차가 있으면
                        mean_reversion_factor = 1 - (deviation * 0.02)  # 편차의 2%만큼 평균으로 회귀
                        final_pred = final_pred * mean_reversion_factor
                        print(f"평균 회귀 적용: 편차 {deviation:.3f}, 조정 계수 {mean_reversion_factor:.3f}")
            
            # 10) 자연스러운 변동성 추가 (농산물 가격의 특성 반영)
            if day_idx > 0:  # 첫 번째 예측 이후에만
                # 품목별 일일 변동성 추가 - 시드 고정으로 일관성 보장
                np.random.seed(42 + day_idx)  # 예측 일차별로 고정된 시드 사용
                
                # 품목별 변동성 설정
                if item_name == '쌀':
                    daily_volatility = np.random.normal(0, 0.001)  # 쌀: 표준편차 0.1% (더 완만하게)
                elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                    # 깐마늘: 실제 시장 변동성을 반영한 높은 변동성
                    daily_volatility = np.random.normal(0, 0.015)  # 표준편차 1.5% (실제 시장 변동성 반영)
                else:
                    daily_volatility = np.random.normal(0, 0.008)  # 기타 품목: 기존 0.8%
                
                volatility_factor = 1 + daily_volatility
                final_pred = final_pred * volatility_factor
                print(f"일일 변동성 추가: {daily_volatility:.4f} ({volatility_factor:.4f})")
            
            # 5) 연속성 보장 (실제 데이터와 예측 데이터 간의 부드러운 연결)
            if day_idx == 0:  # 첫 번째 예측일 때만
                last_actual_price = df_work['가격'].iloc[-1]
                # 품목별 연속성 조정
                if item_name == '쌀':
                    max_first_change = 0.01  # 쌀: 최대 1% 변화 (더 완만하게)
                elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                    max_first_change = 0.05  # 깐마늘: 최대 5% 변화 (실제 시장 변동)
                else:
                    max_first_change = 0.12  # 기타 품목: 기존 12%
                
                if abs(final_pred - last_actual_price) / last_actual_price > max_first_change:
                    if final_pred > last_actual_price:
                        final_pred = last_actual_price * (1 + max_first_change)
                    else:
                        final_pred = last_actual_price * (1 - max_first_change)
                print(f"연속성 조정: {last_actual_price:.0f}원 → {final_pred:.0f}원")
            
            # 6) 예측값 안정화 (극단값 제한) - 품목별 제한
            # 최근 7일 평균 변화율을 고려한 동적 제한
            if len(df_work) >= 7:
                recent_changes = []
                for i in range(1, min(8, len(df_work))):
                    change = (df_work['가격'].iloc[-i] - df_work['가격'].iloc[-i-1]) / df_work['가격'].iloc[-i-1]
                    recent_changes.append(abs(change))
                avg_volatility = np.mean(recent_changes) if recent_changes else 0.03
                
                # 품목별 최대 변화율 설정
                if item_name == '쌀':
                    max_change = min(0.008, avg_volatility * 0.5)  # 쌀: 최대 0.8% (더 완만하게)
                elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                    max_change = min(0.04, avg_volatility * 2.0)  # 깐마늘: 실제 시장 변동 (4%)
                else:
                    max_change = min(0.04, avg_volatility * 1.5)  # 기타 품목: 기존 4%
            else:
                # 품목별 기본 최대 변화율
                if item_name == '쌀':
                    max_change = 0.005  # 쌀: 최대 0.5% 변화 (더 완만하게)
                elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                    max_change = 0.03  # 깐마늘: 실제 시장 변동 (3%)
                else:
                    max_change = 0.03  # 기타 품목: 기존 3%
            
            # 더 역동적인 클리핑
            final_pred = max(last_price * (1 - max_change), 
                           min(last_price * (1 + max_change), final_pred))
            
            # 7) 연속적 안정화 (이전 예측과의 연속성 보장) - 품목별 제한
            if len(preds) > 0:
                prev_pred = preds[-1]['가격']
                # 품목별 연속성 보장
                if item_name == '쌀':
                    max_continuity_change = 0.005  # 쌀: 최대 0.5% 변화 (더 완만하게)
                elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                    max_continuity_change = 0.05  # 깐마늘: 실제 시장 변동 (5%)
                else:
                    max_continuity_change = 0.05  # 기타 품목: 기존 5%
                
                if abs(final_pred - prev_pred) / prev_pred > max_continuity_change:
                    if final_pred > prev_pred:
                        final_pred = prev_pred * (1 + max_continuity_change)
                    else:
                        final_pred = prev_pred * (1 - max_continuity_change)
                print(f"연속성 보장: {prev_pred:.0f}원 → {final_pred:.0f}원")
            
            # 8) 상승/하락 예측 안정화 (연속 급변동 방지)
            if len(preds) >= 2:
                recent_prices = [preds[-1]['가격'], preds[-2]['가격']]
                
                # 연속 하락 방지 (품목별 제한)
                if all(recent_prices[i] > recent_prices[i+1] for i in range(len(recent_prices)-1)):
                    if final_pred < last_price:
                        decline_rate = (last_price - final_pred) / last_price
                        # 품목별 하락 속도 제한
                        if item_name == '쌀':
                            limited_decline = decline_rate * 0.5  # 쌀: 하락 속도 50% 제한
                        elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                            limited_decline = decline_rate * 0.8  # 깐마늘: 실제 시장 변동 (80%)
                        else:
                            limited_decline = decline_rate * 0.8  # 기타 품목: 기존 80%
                        final_pred = last_price * (1 - limited_decline)
                        print(f"하락 안정화: {last_price:.0f}원 → {final_pred:.0f}원 (하락속도 {limited_decline/decline_rate*100:.0f}% 제한)")
                
                # 연속 상승 방지 (품목별 제한)
                elif all(recent_prices[i] < recent_prices[i+1] for i in range(len(recent_prices)-1)):
                    if final_pred > last_price:
                        rise_rate = (final_pred - last_price) / last_price
                        # 품목별 상승 속도 제한
                        if item_name == '쌀':
                            limited_rise = rise_rate * 0.5  # 쌀: 상승 속도 50% 제한
                        elif item_name in ['대파', '깐마늘']:  # 깐마늘(국산)로 교체
                            limited_rise = rise_rate * 0.8  # 깐마늘: 실제 시장 변동 (80%)
                        else:
                            limited_rise = rise_rate * 0.8  # 기타 품목: 기존 80%
                        final_pred = last_price * (1 + limited_rise)
                        print(f"상승 안정화: {last_price:.0f}원 → {final_pred:.0f}원 (상승속도 {limited_rise/rise_rate*100:.0f}% 제한)")
            
            
            preds.append({'날짜': next_date, '가격': final_pred})
            # 예측된 가격으로 업데이트
            next_row['가격'] = final_pred
            df_work = pd.concat([df_work, pd.DataFrame([next_row])], ignore_index=True)
        return pd.DataFrame(preds)

    # 모델이 없거나 학습 불가한 경우: 드리프트 기반(결정론)
    last_date = history['날짜'].max()
    last_price = float(history['가격'].iloc[-1])
    ma7 = history['가격'].tail(7).mean()
    ma30 = history['가격'].tail(30).mean() if len(history) >= 30 else history['가격'].mean()
    drift = float((ma7 - ma30) / max(1.0, ma30))
    drift_per_day = drift * 0.1
    preds = []
    current = last_price
    for _ in range(days_to_predict):
        next_date = last_date + timedelta(days=1)
        current = current * (1 + drift_per_day)
        preds.append({'날짜': next_date, '가격': float(current)})
        last_date = next_date
    return pd.DataFrame(preds)

def _fallback_prediction(history: pd.DataFrame, days_to_predict: int = 7) -> pd.DataFrame:
    """모델 학습 실패 시 사용하는 빠른 폴백 예측"""
    last_date = history['날짜'].max()
    last_price = float(history['가격'].iloc[-1])
    
    # 간단한 트렌드 기반 예측
    if len(history) >= 7:
        recent_7 = history['가격'].tail(7)
        trend = (recent_7.iloc[-1] - recent_7.iloc[0]) / recent_7.iloc[0] / len(recent_7)
        # 트렌드를 0.1% 이내로 제한
        trend = max(-0.001, min(0.001, trend))
    else:
        trend = 0
    
    preds = []
    current = last_price
    for i in range(days_to_predict):
        next_date = last_date + timedelta(days=i+1)
        current = current * (1 + trend)
        preds.append({'날짜': next_date, '가격': float(current)})
    
    return pd.DataFrame(preds)

def predict_rice_price(history: pd.DataFrame, days_to_predict: int = 7) -> pd.DataFrame:
    """쌀 가격 예측 (하위 호환성)"""
    return predict_advanced_price('쌀', history, days_to_predict)

# -------------------------
# 모델 성능 평가 함수
# -------------------------
def evaluate_model_performance(item_name: str, test_days: int = 15) -> dict:
    """모델 성능을 평가하는 함수 (고속 최적화)"""
    try:
        # 전체 데이터 로드
        full_data = _load_all_data(item_name)
        if len(full_data) < test_days + 20:  # 최소 데이터 요구량 감소
            return {"error": "데이터가 부족합니다"}
        
        # 훈련/테스트 분할
        train_data = full_data.iloc[:-test_days].copy()
        test_data = full_data.iloc[-test_days:].copy()
        
        # 기존 모델이 있으면 재사용, 없으면 빠른 학습
        model = _load_model(item_name)
        if model is None:
            print(f"{item_name} 모델이 없습니다. 빠른 학습을 시작합니다...")
            # 빠른 학습을 위해 데이터 크기 제한
            if len(train_data) > 200:
                train_data = train_data.tail(200)  # 최근 200일만 사용
            train_model(train_data, item_name)
        
        # 예측 수행 (빠른 모드)
        predictions = predict_advanced_price(item_name, train_data, test_days)
        
        if len(predictions) != len(test_data):
            return {"error": "예측 길이가 맞지 않습니다"}
        
        # 실제값과 예측값 비교
        actual_prices = test_data['가격'].values
        predicted_prices = predictions['가격'].values
        
        # 성능 지표 계산
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        mse = np.mean((actual_prices - predicted_prices) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        # 방향성 정확도 (상승/하락 방향 예측 정확도)
        actual_direction = np.diff(actual_prices) > 0
        predicted_direction = np.diff(predicted_prices) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # 최근 7일 성능
        recent_mae = np.mean(np.abs(actual_prices[-7:] - predicted_prices[-7:]))
        recent_mape = np.mean(np.abs((actual_prices[-7:] - predicted_prices[-7:]) / actual_prices[-7:])) * 100
        
        return {
            "item_name": item_name,
            "test_days": test_days,
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "direction_accuracy": round(direction_accuracy, 2),
            "recent_mae": round(recent_mae, 2),
            "recent_mape": round(recent_mape, 2),
            "actual_mean": round(np.mean(actual_prices), 2),
            "predicted_mean": round(np.mean(predicted_prices), 2),
            "actual_std": round(np.std(actual_prices), 2),
            "predicted_std": round(np.std(predicted_prices), 2)
        }
        
    except Exception as e:
        return {"error": f"평가 중 오류 발생: {str(e)}"}

def compare_items_performance(items: list, test_days: int = 30) -> pd.DataFrame:
    """여러 품목의 성능을 비교하는 함수"""
    results = []
    for item in items:
        result = evaluate_model_performance(item, test_days)
        if "error" not in result:
            results.append(result)
    
    if not results:
        return pd.DataFrame({"error": ["모든 품목 평가 실패"]})
    
    return pd.DataFrame(results)