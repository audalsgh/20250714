# 16일차

## CNN (Convolutional Neural Network) 용어 정리
1. Convolution (합성곱) : 입력 이미지에 커널(필터)을 슬라이딩하면서 내적 연산을 수행해 특징 맵(feature map)을 만드는 연산.
2. Kernel = Filter (커널 = 필터) : 작은 크기의 가중치 행렬. 합성곱 연산을 통해 입력에서 특정 패턴(엣지, 텍스처 등)을 추출.
3. Stride (스트라이드) : 커널을 이동시킬 때 한 번에 이동하는 픽셀 수. 스트라이드가 클수록 출력 크기가 작아짐.
4. Padding (패딩) : 입력 주변에 0(또는 다른 값)을 추가하여 출력 크기를 조절하거나 경계 정보 손실을 방지.
5. Activation Function (활성화 함수, 특히 ReLU) : 합성곱 또는 완전연결 층의 선형 출력에 비선형성을 부여. 대표적으로 ReLU, Sigmoid, Tanh 등이 있음.
6. Pooling (풀링, 특히 Max Pooling) : 특징 맵의 크기를 줄여 연산량을 줄이고, 위치 변동에 강인성을 부여.<br>Max Pooling: 영역 내 최대값 선택<br>Average Pooling: 영역 내 평균값 선택
7. Fully Connected Layer (완전연결 층) : 모든 뉴런이 이전 층의 모든 뉴런과 연결된 층. 분류 및 회귀를 위해 사용.
8. Loss Function (손실 함수) : 모델 예측값과 실제값 간 차이를 수치화. 대표적으로 분류에는 Cross-Entropy, 회귀에는 MSE 사용.
9. Optimizer (최적화 알고리즘) : 손실 함수를 최소화하도록 파라미터를 업데이트하는 알고리즘.<br>SGD (Stochastic Gradient Descent), Momentum, RMSprop, Adam 등
10. Epoch (에폭) : 전체 학습 데이터를 한 번 모두 사용해 파라미터를 업데이트한 횟수.
11. Batch (배치) : 한 번에 신경망에 입력으로 넣어 학습시키는 샘플 묶음. 배치 크기에 따라 학습 안정성과 속도가 달라짐.
12. Regularization (정규화) : 모델 복잡도를 제어해 과적합을 방지하는 기법. L1/L2 페널티, Dropout 등이 있음.
