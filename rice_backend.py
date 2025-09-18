import os
from datetime import timedelta
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except Exception:
    xgb = None


# -------------------------
# 데이터 로드 및 전처리
# -------------------------
def _read_rice_history_csv(csv_path: str = 'rice.csv') -> pd.DataFrame:
    encodings_to_try = ['utf-8-sig', 'utf-8', 'cp949', 'ISO-8859-1']
    df = None
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc, encoding_errors='replace')
            if '날짜' in df.columns and ('가격(20kg)' in df.columns or '가격' in df.columns):
                break
        except Exception as e:
            last_error = e
            continue
    if df is None or '날짜' not in df.columns:
        raise FileNotFoundError(f"{csv_path} 파일을 읽을 수 없습니다. 마지막 오류: {last_error}")
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    price_col = '가격(20kg)' if '가격(20kg)' in df.columns else '가격'
    price_history = df.groupby('날짜')[price_col].mean().reset_index().dropna(subset=['날짜'])
    price_history = price_history.sort_values('날짜')
    price_history.rename(columns={price_col: '가격'}, inplace=True)
    price_history['가격'] = pd.to_numeric(price_history['가격'], errors='coerce')
    price_history = price_history.dropna(subset=['가격'])
    return price_history.reset_index(drop=True)


def get_rice_history(days: int = 365) -> pd.DataFrame:
    try:
        hist = _read_rice_history_csv()
    except Exception:
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
        prices = np.full(days, 52000.0)
        return pd.DataFrame({'날짜': pd.to_datetime(dates), '가격': prices})
    if days is not None and days > 0:
        hist = hist.tail(days)
    return hist.reset_index(drop=True)


# -------------------------
# 피처 엔지니어링(노트북 이식)
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
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df['가격'].shift(lag)
    df['ma_7'] = df['가격'].rolling(7, min_periods=1).mean()
    df['ma_30'] = df['가격'].rolling(30, min_periods=1).mean()
    return df


def _build_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _add_time_features(df)
    df = _add_lag_rolling(df)
    df['target_next'] = df['가격'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ['target_next', '날짜'] and c != '가격']
    return df, feature_cols


# -------------------------
# 모델 학습/저장/로드
# -------------------------
MODEL_PATH = 'xgb_model.json'


def train_model(history: pd.DataFrame) -> None:
    if xgb is None:
        return
    df_sup, feature_cols = _build_supervised(history)
    if len(df_sup) < 50:
        return
    X = df_sup[feature_cols]
    y = df_sup['target_next']
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y, verbose=False)
    model.save_model(MODEL_PATH)


def _load_model():
    if xgb is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    m = xgb.XGBRegressor()
    m.load_model(MODEL_PATH)
    return m


# -------------------------
# 재귀적 예측
# -------------------------
def predict_rice_price(history: pd.DataFrame, days_to_predict: int = 14) -> pd.DataFrame:
    history = history.sort_values('날짜').reset_index(drop=True)

    model = _load_model()
    if model is None:
        train_model(history)
        model = _load_model()

    if model is not None:
        df_work = history.copy()
        preds = []
        for _ in range(days_to_predict):
            next_date = df_work['날짜'].iloc[-1] + timedelta(days=1)
            tmp = pd.concat([df_work, pd.DataFrame({'날짜': [next_date], '가격': [df_work['가격'].iloc[-1]]})], ignore_index=True)
            tmp = _add_time_features(tmp)
            tmp = _add_lag_rolling(tmp)
            feature_cols = [c for c in tmp.columns if c not in ['날짜', '가격']]
            x_next = tmp.iloc[[-1]][feature_cols]
            y_hat = float(model.predict(x_next)[0])
            preds.append({'날짜': next_date, '가격': y_hat})
            df_work = pd.concat([df_work, pd.DataFrame({'날짜': [next_date], '가격': [y_hat]})], ignore_index=True)
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


