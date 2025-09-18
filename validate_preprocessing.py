import pandas as pd
import numpy as np

print("[validate] Start")

# --- 1. 데이터 로드 (노트북과 동일 경로 가정) ---
df_exchange = pd.read_csv('exchange_rate.csv')
df_oil = pd.read_csv('oil.csv', encoding='cp949')
df_rice = pd.read_excel('감자_도매_데이터.xlsx')
df_weather = pd.read_csv('top_weather_features.csv')

# '누적강수량' 컬럼 존재 시 제거
if '누적강수량' in df_weather.columns:
    df_weather.drop('누적강수량', axis=1, inplace=True)

# --- 2. 개별 데이터 전처리 (노트북의 새 로직과 동일) ---

# 1) 날씨 데이터
_dfw = df_weather.copy()
_dfw.columns = [str(c).strip() for c in _dfw.columns]
name_map = {
    'date': '날짜', '일자': '날짜', '날짜': '날짜',
    '누적평균기온': '누적평균기온', '평균기온': '누적평균기온', '기온': '누적평균기온',
    '누적일조합': '누적일조합', '일조합': '누적일조합', '일조시간': '누적일조합'
}
_dfw = _dfw.rename(columns={c: name_map.get(c, c) for c in _dfw.columns})
if '날짜' not in _dfw.columns:
    _dfw.rename(columns={_dfw.columns[0]: '날짜'}, inplace=True)
if '누적평균기온' not in _dfw.columns:
    _dfw['누적평균기온'] = np.nan
if '누적일조합' not in _dfw.columns:
    _dfw['누적일조합'] = np.nan
_dfw['날짜'] = pd.to_datetime(_dfw['날짜'], errors='coerce')
if _dfw.index.name == '날짜' or ('날짜' in (list(_dfw.index.names) if _dfw.index.names is not None else [])):
    _dfw = _dfw.reset_index()
_dfw = _dfw.loc[:, ~_dfw.columns.duplicated()]
if '날짜' not in _dfw.columns:
    raise ValueError("날씨 데이터에 '날짜' 컬럼을 만들지 못했습니다.")
df_weather = (_dfw.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True))

# 2) 쌀 데이터
_df_rice = df_rice.copy()
_df_rice.columns = [str(c).strip() for c in _df_rice.columns]
name_map_rice = {
    '일자': '날짜', 'date': '날짜', '날짜': '날짜',
    '가격': '가격', '평균가격': '가격', '도매가격': '가격', '소매가격': '가격'
}
_df_rice = _df_rice.rename(columns={c: name_map_rice.get(c, c) for c in _df_rice.columns})
for col in ['품목명','품종명','시장명','지역명']:
    if col in _df_rice.columns:
        _df_rice.drop(columns=[col], inplace=True)
if '날짜' not in _df_rice.columns:
    _df_rice.rename(columns={_df_rice.columns[0]: '날짜'}, inplace=True)
if '가격' not in _df_rice.columns:
    num_cols = _df_rice.select_dtypes(include='number').columns.tolist()
    if num_cols:
        _df_rice.rename(columns={num_cols[0]: '가격'}, inplace=True)
    else:
        _df_rice['가격'] = np.nan
_df_rice['날짜'] = pd.to_datetime(_df_rice['날짜'], errors='coerce')
_df_rice['가격'] = pd.to_numeric(_df_rice['가격'], errors='coerce')
df_rice = (_df_rice.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True))

# 3) 환율
_df_exchange = df_exchange.copy()
_df_exchange.columns = [str(c).strip() for c in _df_exchange.columns]
if '종가' in _df_exchange.columns and '환율' not in _df_exchange.columns:
    _df_exchange.rename(columns={'종가': '환율'}, inplace=True)
if '날짜' not in _df_exchange.columns:
    _df_exchange.rename(columns={_df_exchange.columns[0]: '날짜'}, inplace=True)
_df_exchange['날짜'] = pd.to_datetime(_df_exchange['날짜'], errors='coerce')
if '환율' in _df_exchange.columns:
    _df_exchange['환율'] = pd.to_numeric(_df_exchange['환율'].astype(str).str.replace(',', ''), errors='coerce')
df_exchange = (_df_exchange.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True))

# 4) 유가
_df_oil = df_oil.copy()
_df_oil.columns = [str(c).strip() for c in _df_oil.columns]
if len(_df_oil.columns) >= 2:
    _df_oil = _df_oil.rename(columns={_df_oil.columns[0]: '날짜', _df_oil.columns[1]: '유가'})
if '날짜' not in _df_oil.columns:
    _df_oil.rename(columns={_df_oil.columns[0]: '날짜'}, inplace=True)
_df_oil['날짜'] = pd.to_datetime(_df_oil['날짜'], errors='coerce')
df_oil = (_df_oil.dropna(subset=['날짜']).groupby('날짜', as_index=False).mean(numeric_only=True))

# --- 3. 병합 테스트 ---
new = pd.merge(df_exchange, df_oil, how='inner', on='날짜')
new2 = pd.merge(new, df_weather, how='inner', on='날짜')
df = pd.merge(new2, df_rice, how='inner', on='날짜')

print("[validate] Shapes:")
print("  weather:", df_weather.shape)
print("  exchange:", df_exchange.shape)
print("  oil:", df_oil.shape)
print("  rice:", df_rice.shape)
print("  merged df:", df.shape)
print("  merged cols:", df.columns.tolist())
print("[validate] Done")


