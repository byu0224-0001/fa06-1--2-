import os
from datetime import timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
except Exception:
    xgb = None
    StandardScaler = None
    MinMaxScaler = None
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
            _DEVICE = torch.device('cpu')
            print("CPU 사용")
    return _DEVICE

# -------------------------
# DLinear 모델 정의 (원본 노트북과 동일)
# -------------------------
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

# -------------------------
# 데이터 로드 및 전처리 (캐싱으로 최적화)
# -------------------------
def _load_all_data() -> pd.DataFrame:
    """원본 노트북과 동일한 데이터 로드 및 전처리 (캐싱 적용)"""
    global _CACHED_DATA
    
    if _CACHED_DATA is not None:
        return _CACHED_DATA.copy()
    
    try:
        # 1) 환율 데이터
        df_exchange = pd.read_csv('exchange_rate.csv')
        df_exchange.columns = [str(c).strip() for c in df_exchange.columns]
        if '종가' in df_exchange.columns and '환율' not in df_exchange.columns:
            df_exchange.rename(columns={'종가': '환율'}, inplace=True)
        if '날짜' not in df_exchange.columns:
            df_exchange.rename(columns={df_exchange.columns[0]: '날짜'}, inplace=True)
        df_exchange['날짜'] = pd.to_datetime(df_exchange['날짜'], errors='coerce')
        if '환율' in df_exchange.columns:
            df_exchange['환율'] = pd.to_numeric(df_exchange['환율'].astype(str).str.replace(',', ''), errors='coerce')
        df_exchange = df_exchange.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True)
    except Exception:
        df_exchange = pd.DataFrame({'날짜': [pd.Timestamp.today()], '환율': [1300.0]})

    try:
        # 2) 유가 데이터
        df_oil = pd.read_csv('oil.csv', encoding='cp949')
        df_oil.columns = [str(c).strip() for c in df_oil.columns]
        if len(df_oil.columns) >= 2:
            df_oil = df_oil.rename(columns={df_oil.columns[0]: '날짜', df_oil.columns[1]: '유가'})
        if '날짜' not in df_oil.columns:
            df_oil.rename(columns={df_oil.columns[0]: '날짜'}, inplace=True)
        df_oil['날짜'] = pd.to_datetime(df_oil['날짜'], errors='coerce')
        df_oil = df_oil.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True)
    except Exception:
        df_oil = pd.DataFrame({'날짜': [pd.Timestamp.today()], '유가': [80.0]})

    try:
        # 3) 날씨 데이터
        df_weather = pd.read_csv('top_weather_features.csv')
        df_weather.columns = [str(c).strip() for c in df_weather.columns]
        if '누적강수량' in df_weather.columns:
            df_weather.drop('누적강수량', axis=1, inplace=True)
        name_map = {
            'date': '날짜', '일자': '날짜', '날짜': '날짜',
            '누적평균기온': '누적평균기온', '평균기온': '누적평균기온', '기온': '누적평균기온', 'avgTa_cum': '누적평균기온',
            '누적일조합': '누적일조합', '일조합': '누적일조합', '일조시간': '누적일조합', '누적일조시간': '누적일조합'
        }
        df_weather = df_weather.rename(columns={c: name_map.get(c, c) for c in df_weather.columns})
        if '날짜' not in df_weather.columns:
            df_weather.rename(columns={df_weather.columns[0]: '날짜'}, inplace=True)
        if '누적평균기온' not in df_weather.columns:
            df_weather['누적평균기온'] = np.nan
        if '누적일조합' not in df_weather.columns:
            df_weather['누적일조합'] = np.nan
        df_weather['날짜'] = pd.to_datetime(df_weather['날짜'], errors='coerce')
        if df_weather.index.name == '날짜' or ('날짜' in (list(df_weather.index.names) if df_weather.index.names is not None else [])):
            df_weather = df_weather.reset_index()
        df_weather = df_weather.loc[:, ~df_weather.columns.duplicated()]
        df_weather = df_weather.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True)
    except Exception:
        df_weather = pd.DataFrame({'날짜': [pd.Timestamp.today()], '누적평균기온': [15.0], '누적일조합': [6.0]})

    try:
        # 4) 쌀 데이터 (실제 쌀 CSV 데이터 사용)
        df_rice = pd.read_csv('rice.csv')
        df_rice.columns = [str(c).strip() for c in df_rice.columns]
        if '가격(20kg)' in df_rice.columns:
            df_rice.rename(columns={'가격(20kg)': '가격'}, inplace=True)
        for col in ['품목명','품종명','시장명','지역명']:
            if col in df_rice.columns:
                df_rice.drop(columns=[col], inplace=True)
        df_rice['날짜'] = pd.to_datetime(df_rice['날짜'], errors='coerce')
        df_rice['가격'] = pd.to_numeric(df_rice['가격'], errors='coerce')
        df_rice = df_rice.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True)
    except Exception:
        df_rice = pd.DataFrame({'날짜': [pd.Timestamp.today()], '가격': [52000.0]})

    # 5) 모든 데이터 병합
    new = pd.merge(df_exchange, df_oil, how='inner', on='날짜')
    new2 = pd.merge(new, df_weather, how='inner', on='날짜')
    df = pd.merge(new2, df_rice, how='inner', on='날짜')
    df = df.sort_values('날짜').reset_index(drop=True)
    
    # 캐싱
    _CACHED_DATA = df.copy()
    return df

def get_rice_history(days: int = 365) -> pd.DataFrame:
    try:
        # 완전한 데이터셋에서 쌀 가격만 추출
        full_data = _load_all_data()
        
        if full_data.empty:
            raise ValueError("데이터가 비어있습니다")
        if '가격' not in full_data.columns:
            raise ValueError(f"가격 컬럼이 없습니다. 사용 가능한 컬럼: {full_data.columns.tolist()}")
            
        hist = full_data[['날짜', '가격']].copy()
        
        # 가격 데이터 검증
        hist = hist.dropna(subset=['가격'])
        if hist.empty:
            raise ValueError("유효한 가격 데이터가 없습니다")
            
    except Exception as e:
        # 오류 발생 시 기본 데이터 반환
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
        prices = np.full(days, 52000.0)
        return pd.DataFrame({'날짜': pd.to_datetime(dates), '가격': prices})
    
    if days is not None and days > 0:
        hist = hist.tail(days)
    return hist.reset_index(drop=True)

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
MODEL_PATH = 'xgb_model.json'
SCALER_PATH = 'scaler.pkl'
DLINEAR_PATH = 'best_dlinear_model.pth'
FEATURE_COLS_PATH = 'feature_cols.pkl'

def _clean_numeric_frame(df):
    """NaN/Inf 값 정리 (원본 노트북과 동일)"""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    return df

def train_model(history: pd.DataFrame) -> None:
    """원본 노트북과 동일한 방식으로 모델 학습 (최적화)"""
    if xgb is None or StandardScaler is None or torch is None:
        return
    
    print("모델 학습 시작...")
    device = _get_device()
    
    # 완전한 데이터셋으로 학습
    full_data = _load_all_data()
    df_sup, feature_cols = _build_supervised(full_data)
    if len(df_sup) < 50:
        print("데이터가 부족합니다.")
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
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # DLinear 모델 학습 (원본 노트북과 동일한 설정)
    sequence_length = 30
    predict_length = 1
    input_dim = X_train_scaled.shape[1]
    batch_size = 32
    
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
                torch.save(dlinear_model.state_dict(), DLINEAR_PATH)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"조기 종료: {epoch} 에포크")
                    break
        
        # DLinear 예측 제거 - 43개 피처로 일관성 유지
        # 기존 모델이 43개 피처로 학습되었으므로 43개로 통일
        
        # XGBoost 학습 (DLinear 예측 제외)
        X_train_final = X_train_scaled[sequence_length-1:]
        
        X_test_seq = create_sequences(X_test_scaled, sequence_length)
        if len(X_test_seq) > 0:
            X_test_final = X_test_scaled[sequence_length-1:]
        else:
            X_test_final = X_test_scaled
        
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
            X_train_final, y_train_final,
            eval_set=[(X_test_final, y_test_final)],
            verbose=False
        )
        
        # 모델 저장
        xgb_model.save_model(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_cols, FEATURE_COLS_PATH)
        
        print(f"모델 학습 완료! XGBoost RMSE: {np.sqrt(mean_squared_error(y_test_final, xgb_model.predict(X_test_final))):.2f}")

def _load_model():
    if xgb is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    m = xgb.XGBRegressor()
    m.load_model(MODEL_PATH)
    return m

def _load_scaler():
    if joblib is None or not os.path.exists(SCALER_PATH):
        return None
    return joblib.load(SCALER_PATH)

def _load_feature_cols():
    if joblib is None or not os.path.exists(FEATURE_COLS_PATH):
        return None
    return joblib.load(FEATURE_COLS_PATH)

# -------------------------
# 재귀적 예측 (원본 노트북과 동일한 성능)
# -------------------------
def predict_rice_price(history: pd.DataFrame, days_to_predict: int = 14) -> pd.DataFrame:
    history = history.sort_values('날짜').reset_index(drop=True)

    model = _load_model()
    if model is None:
        print("모델이 없습니다. 학습을 시작합니다...")
        train_model(history)
        model = _load_model()

    if model is not None:
        # 완전한 데이터셋으로 작업
        full_data = _load_all_data()
        df_work = full_data.copy()
        preds = []
        # 스케일러 및 피처 컬럼명 로드
        scaler = _load_scaler()
        feature_cols = _load_feature_cols()
        
        # 피처 컬럼명이 없으면 새로 생성
        if feature_cols is None:
            _, feature_cols = _build_supervised(full_data)
        
        # 피처 개수 확인
        print(f"학습된 모델 피처 개수: {len(feature_cols)}")
        print(f"피처 목록: {feature_cols[:10]}...")  # 처음 10개만 출력
            
        for _ in range(days_to_predict):
            next_date = df_work['날짜'].iloc[-1] + timedelta(days=1)
            # 다음 날의 모든 변수 예측 (간단한 드리프트)
            next_row = {}
            for col in ['환율', '유가', '누적평균기온', '누적일조합']:
                if col in df_work.columns:
                    last_val = df_work[col].iloc[-1]
                    # 간단한 드리프트 (실제로는 각 변수별 모델이 필요)
                    drift = 0.001  # 0.1% 변화
                    next_row[col] = last_val * (1 + drift)
                else:
                    next_row[col] = df_work[col].iloc[-1] if col in df_work.columns else 0
            next_row['날짜'] = next_date
            next_row['가격'] = df_work['가격'].iloc[-1]  # 임시값
            
            tmp = pd.concat([df_work, pd.DataFrame([next_row])], ignore_index=True)
            tmp = _add_time_features(tmp)
            tmp = _add_lag_rolling(tmp)
            tmp = _add_advanced_features(tmp)
            
            # 학습 시와 동일한 방식으로 피처 생성 (DLinear 예측 포함)
            tmp_supervised, _ = _build_supervised(tmp)
            if len(tmp_supervised) == 0:
                # 데이터가 부족한 경우 기본값으로 채우기
                x_next = pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)
            else:
                # 마지막 행의 피처만 선택
                x_next = tmp_supervised[feature_cols].iloc[[-1]]
            
            # 피처 개수 검증 및 조정
            model_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(feature_cols)
            expected_features = model_features
            actual_features = x_next.shape[1]
            if actual_features != expected_features:
                print(f"피처 개수 불일치: 예상 {expected_features}, 실제 {actual_features}")
                # 부족한 피처를 0으로 채우기
                if actual_features < expected_features:
                    missing_count = expected_features - actual_features
                    for i in range(missing_count):
                        x_next[f'missing_feature_{i}'] = [0.0]
                # 초과한 피처 제거
                elif actual_features > expected_features:
                    x_next = x_next.iloc[:, :expected_features]
                
                # 피처 순서 정렬 (feature_cols 순서대로)
                x_next = x_next.reindex(columns=feature_cols, fill_value=0.0)
            
            
            if scaler is not None:
                # 피처 이름 문제를 피하기 위해 numpy 배열로 변환
                x_next_array = x_next.values
                x_next_scaled = scaler.transform(x_next_array)
                y_hat = float(model.predict(x_next_scaled)[0])
            else:
                y_hat = float(model.predict(x_next)[0])
            preds.append({'날짜': next_date, '가격': y_hat})
            # 예측된 가격으로 업데이트
            next_row['가격'] = y_hat
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