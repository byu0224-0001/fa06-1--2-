from rice_backend import get_rice_history, predict_rice_price

print("=== 최적화된 백엔드 테스트 ===")
hist = get_rice_history(365)
print(f"쌀 데이터 shape: {hist.shape}")
print(f"최근 가격: {hist['가격'].iloc[-1]:.0f}원")

print("\n예측 시작...")
pred = predict_rice_price(hist, 7)
print(f"예측 결과: {pred['가격'].iloc[-1]:.0f}원")
print("테스트 완료!")

