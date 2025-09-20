# 사장님을 위한 AI 식자재 비서

농산물 가격 예측 및 구매 타이밍 분석을 제공하는 AI 기반 대시보드입니다.

## 주요 기능

- **실시간 가격 예측**: 쌀, 깐마늘, 양파 등 주요 농산물의 가격 예측
- **AI 구매 팁**: OpenAI GPT-4o를 활용한 스마트한 구매 타이밍 분석
- **다양한 예측 기간**: 3일, 7일, 15일 예측 제공
- **시각화**: Plotly를 활용한 인터랙티브 그래프

## 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 설정
`.streamlit/secrets.toml` 파일을 생성하고 OpenAI API 키를 설정하세요:
```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

### 5. 애플리케이션 실행
```bash
streamlit run "2차_프로젝트_대시보드.py"
```

## 프로젝트 구조

```
├── 2차_프로젝트_대시보드.py    # Streamlit 프론트엔드
├── rice_backend.py            # 백엔드 로직 및 예측 모델
├── requirements.txt           # Python 의존성
├── .streamlit/
│   └── secrets.toml          # API 키 설정
├── data/                     # 데이터 파일
│   ├── exchange_rate.csv
│   ├── oil.csv
│   ├── rice.csv
│   ├── top_weather_features.csv
│   └── *_도매_데이터.xlsx
├── models/                   # 학습된 모델 파일
│   ├── *_dlinear_model.pth
│   ├── *_feature_cols.pkl
│   ├── *_scaler.pkl
│   └── *_xgb_model.json
└── README.md
```

## 사용된 기술

- **Frontend**: Streamlit
- **Backend**: Python, Pandas, NumPy
- **ML Models**: DLinear, XGBoost
- **Visualization**: Plotly
- **AI Analysis**: OpenAI GPT-4o

## 주요 품목

- 쌀 (20kg)
- 깐마늘 (20kg) 
- 양파 (15kg)

## 🚀 Streamlit Cloud 배포

### 1. Streamlit Cloud에 배포하기

1. [Streamlit Cloud](https://share.streamlit.io/)에 접속하여 GitHub 계정으로 로그인
2. **New app** 버튼 클릭
3. Repository 선택 및 메인 파일 경로 설정: `2차_프로젝트_대시보드.py`
4. **Deploy!** 클릭

### 2. API 키 설정

배포 후 **Settings** → **Secrets**에서 다음 내용을 추가:

```toml
OPENAI_API_KEY = "your-actual-openai-api-key"
```

### 3. 환경 설정 확인

- ✅ `.streamlit/secrets.toml`은 `.gitignore`에 포함되어 GitHub에 업로드되지 않습니다
- ✅ `secrets.toml.template` 파일을 참고하여 로컬 개발 시 API 키를 설정하세요

## 🔧 문제 해결

### AI 구매 팁이 작동하지 않는 경우
1. Streamlit Cloud의 **Secrets** 설정 확인
2. API 키가 올바르게 설정되었는지 확인
3. OpenAI API 사용량 및 크레딧 확인

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.