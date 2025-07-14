# 16일차

## CNN (Convolutional Neural Network) 정리
왜 중요한가?<br>
입력 이미지에 커널을 슬라이딩하며 국소 영역의 정보를 요약(내적)해 새로운 특징 맵을 만듬. <br>전반의 패턴(엣지, 코너, 질감 등)을 ‘위치-불변성’ 있게 학습할 수 있어, 복잡한 패턴도 단계별로 계층화해 인식 가능<br>
스트라이드와 패딩 설정으로 출력 맵의 공간 해상도(크기)를 자유롭게 설정가능.<br>
완전연결 층에선 모든 입력 뉴런과 출력 뉴런이 연결되있어 ‘전역적’ 특징을 학습하지만, 파라미터 수가 많아 과적합 위험도 존재.

**요약: CNN 학습 흐름
입력 -> (Conv → Activation → Pooling) × N = 계층별로 국소 특징 추출
Flatten -> Fully Connected -> Activation -> Loss 계산 -> Optimizer로 파라미터 업데이트 -> Epoch 반복 + Regularization 적용**


1. Convolution (합성곱) : 입력 이미지에 커널(필터)을 슬라이딩하면서 내적 연산을 수행해 특징 맵(feature map)을 만드는 연산.
2. Kernel = Filter (커널 = 필터) : 3×3, 5×5 같이 작은 크기의 가중치 행렬. 각 국소 영역과 곱해져 입력에서 특정 패턴(엣지, 텍스처 등)을 추출.
3. Stride (스트라이드) : 커널을 이동시킬 때 한 번에 이동하는 픽셀 수.<br>Stride=1: 한 픽셀씩, Stride=2: 두 픽셀씩 이동 → 출력 맵 크기가 달라짐.
4. Padding (패딩) : 입력 주변에 0(또는 다른 값)을 추가하여 출력 크기를 조절하거나 경계 정보 손실을 방지.<br>‘same’ 패딩: 출력 크기를 입력과 같게, ‘valid’ 패딩: 추가 없이 순수 합성곱만 수행.
5. Activation Function (활성화 함수, 특히 ReLU) : 합성곱 또는 완전연결 층의 선형 출력에 비선형성을 부여해, 네트워크가 복잡한 함수의 근사를 가능하게 함. 대표적으로 ReLU, Sigmoid, Tanh 등이 있음.<br>ReLU: 음수는 0, 양수는 그대로 → 경사소실 문제 완화
6. Pooling (풀링, 특히 Max Pooling) : 특징 맵의 크기를 줄여 연산량을 줄이고, 위치 변동에 강인성을 부여.<br>Max Pooling: 영역 내 최대값 선택<br>Average Pooling: 영역 내 평균값 선택<br>계산량 절감 + 오버피팅 완화 + 위치 약간의 변화 불변성 확보
7. Fully Connected Layer (완전연결 층) : 합성곱·풀링 과정을 거쳐 추출된 ‘글로벌 특징’을 종합해 최종 클래스 점수나 연속값을 예측하는 층<br>평탄화(Flatten)된 벡터를 입력으로 받아, 최종 분류(class scores)나 회귀값을 출력.
8. Epoch (에폭) : 전체 학습 데이터를 한 번 모두 사용해 파라미터를 업데이트한 횟수.
9. Batch (배치) : 한 번에 신경망에 입력으로 넣어 학습시키는 샘플 묶음. 배치 크기에 따라 학습 안정성과 속도가 달라짐.
10. Loss Function (손실 함수) : 모델 예측값과 실제값 간 차이를 수치화. 대표적으로 분류에는 Cross-Entropy, 회귀에는 MSE 사용.
11. Optimizer (최적화 알고리즘) : 손실 함수를 최소화하도록 파라미터를 업데이트하는 알고리즘.<br>ex) SGD (Stochastic Gradient Descent), Momentum, RMSprop, Adam 등
12. Regularization (정규화) : 모델 복잡도를 제어해 과적합을 방지하는 기법.<br>L1/L2 페널티 = 가중치 크기를 페널티로 추가<br>Dropout = 학습 시 뉴런 일부를 무작위 비활성화 등이 있음.
