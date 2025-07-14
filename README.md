# 16일차

## 1. CNN (Convolutional Neural Network) 필터 종류 정리
https://claude.ai/public/artifacts/2cebc728-66b5-414a-9e97-991f60a2a7e1<br>
컨볼루션에 대한 애니메이션

<img width="1112" height="451" alt="image" src="https://github.com/user-attachments/assets/40de01f9-e85c-4733-9d86-febbf3906741" />
- 대각선 필터로 컨볼루션해나가는 모습, stride=1 이기에 (3x3) 피쳐맵이 나온다!<br>
- 피쳐맵 결과에서 값이 클수록 픽셀이 많이 검출된 것!<br>
<br>
https://claude.ai/public/artifacts/a3bda456-4c3f-4127-a921-21ad4c351c98<br>
이미지를 직접 업로드하여 4개의 필터를 사용해보는 애니메이션 링크 
<img width="985" height="638" alt="image" src="https://github.com/user-attachments/assets/d0e9361c-ed1b-41d3-95a0-650b26084950" />
<img width="1566" height="529" alt="image" src="https://github.com/user-attachments/assets/ea0e485a-4bdb-4bf4-a460-e4f2695d6f7e" /><br>
- 왼쪽 열이 음수값인 수직 엣지 필터 사용시, 왼쪽에서 오른쪽으로 밝기가 증가하는 엣지를 검출함<br>
- 피쳐맵 결과의 양수값 = 왼쪽 -> 오른쪽이 밝아지는 엣지 존재 / 피쳐맵 결과의 음수값 = 왼쪽 -> 오른쪽이 어두워지는 엣지 존재
<img width="1573" height="530" alt="image" src="https://github.com/user-attachments/assets/ea708832-5207-4c44-a152-0d10411c80a4" />
- 위쪽 행이 음수값이 수평 엣지 필터 사용시, 위에서 아래쪽으로 밝기가 증가하는 엣지를 검출함<br>
- 피쳐맵 결과의 양수값 = 위쪽 -> 아래쪽이 밝아지는 엣지 존재 / 피쳐맵 결과의 음수값 = 왼쪽 -> 오른쪽이 어두워지는 엣지 존재
<img width="1568" height="529" alt="image" src="https://github.com/user-attachments/assets/94b7580a-f78c-437a-bfda-c26bb8790462" />
- 가중치가 0.11인 블러 필터 사용시, 이미지가 전반적으로 부드럽게, 흐릿해보이게 함<br>
- 노이즈·세부 표현 감소 : 갑작스러운 밝기 변화(고주파 성분)를 부드럽게 해, 엣지나 디테일이 희미해지며, 경계선 주변이 자연스럽게 그라데이션으로 변함<br>
- 시각적 부드러움 향상 : 얼굴 보정, 배경 흐림 등에서 활용됨<br>
- 계산법 : 주변 픽셀들의 값을 9개를 모두 더하고 9로 나눠 평균내거나, 비슷한 가중치로 섞어서 중앙 픽셀 값을 대체하는 저역 통과 필터(low-pass)<br>
- 피쳐맵 결과의 값 = 절댓값이 클수록 주변 픽셀과 혼합될 때 상대적으로 큰 변화가 일어났음을 의미하고,<br>
   주변보다 특히 밝았기에 블러효과가 많이 일어났다고 판단 가능
<img width="1575" height="540" alt="image" src="https://github.com/user-attachments/assets/72d31710-2dad-4427-8420-1f11deeaa716" />
- 중심 픽셀 값에 5를 곱하면서, 주변 픽셀을 -1을 곱해 빼는 방식으로, 중심 픽셀이 주변보다 상대적으로 매우 밝음을 강조하는 샤프닝 필터를 사용함.<br>
- 피쳐맵 결과의 값 = 절댓값이 클수록 이미지의 경계(엣지)와 세부 디테일이 부각됨<br>
<br>

https://claude.ai/public/artifacts/df7a5986-dd0a-4a16-af85-ad90959de392<br>
패딩과 렐루 애니메이션 링크
<img width="1231" height="679" alt="image" src="https://github.com/user-attachments/assets/4b4127b0-6865-4063-8a8f-d57ab0bec276" />
- 일반적으로 이미지 픽셀마다 RGB값 3개를 모두 저장한다.
<img width="1186" height="593" alt="image" src="https://github.com/user-attachments/assets/d1cdf3a2-cb21-46a1-9abb-b37bfead7ec3" />
- 패딩 개념 : (5x5) 입력인데, 필터를 거치니 (3x3) 결과를 얻었다. 원래 크기 (5x5)를 다시 얻기위해 바깥 테두리에 1칸씩 늘리고 필터와 컨볼루션을 하자!
<img width="1372" height="461" alt="image" src="https://github.com/user-attachments/assets/a3296157-cd73-4e53-9d0a-8cef6889cd2d" />
- 컨볼루션을 거치면서 0값이였던 테두리도 값을 가지게 된다.<br>
<img width="1383" height="650" alt="image" src="https://github.com/user-attachments/assets/9a6f3244-520e-4ce0-a185-127dcfc5727d" />
- ReLU(렐루) : ReLU(x) = max(0, x)를 입력해 음수는 모두 0이 되므로, 활성화된 뉴런(양수값)이 상대적으로 적어, 효율적이고 안정적으로 빠른 표현 학습이 가능.
<img width="1383" height="491" alt="image" src="https://github.com/user-attachments/assets/c80bdee4-a996-44ed-9077-3774104e962a" />
- 렐루를 거쳐서 훨씬 빠르고 안정적이게 컨볼루션을 진행했고, 그 피쳐맵을 통해 결과 이미지(히트맵)을 얻어냈다.<br>
- 히트맵 : 피쳐맵의 각 RGB값에 색을 입혀 "모델이 중요하게 판단한 부분이 어딘지 강도(strength)"를 한눈에 보여 주는 시각화 기법.<br>
- 빨간색 = 모델이 중요하다고 판단하여 피쳐가 강하게 반응한 부분<br>
- 파란색 = 모델이 덜 중요하다고 판단한 부분.

## 2. CNN 세부 용어 정리
왜 중요한가?<br>
흐름 : "입력 이미지"에 "커널(필터)"를 슬라이딩하며 국소 영역의 정보를 요약(컨볼루션) -> 새로운 피쳐맵 결과를 얻음.<br>
특징<br>
- 전반적인 패턴(엣지, 코너, 질감 등)을 ‘위치-불변성’ 있게 학습할 수 있어, 복잡한 패턴도 단계별로 계층화해 인식 가능<br>
- stride와 패딩 설정으로 출력되는 피쳐맵의 공간 해상도(크기)를 자유롭게 설정가능.<br>
- 완전연결 층에선 모든 입력 뉴런과 출력 뉴런이 연결되있어 ‘전역적’ 특징을 학습하지만, 파라미터 수가 많아 과적합 위험도 존재.<br>
- 실제 상황에서는 입력과 정답 사이의 관계가 단순 직선으로 설명되지 않는다.<br>
- ReLU는 입력이 음수면 0, 양수면 그대로 통과시키는 “꺾인” 형태의 선형이 아닌(non-linear) 함수.<br>
- 각 층 다음에 넣어 주면 층들을 조합했을 때 결과가 복잡한 곡선이 될 수 있다 = 다양한 모양의 결정 경계나 입력-출력 관계를 모델이 학습할 수 있게 됨.<br>
<br>

**요약 : CNN 학습 흐름<br>
입력 -> (Convolution -> Kernel) = 계층별로 국소 특징 추출 ->  CNN에서 합성곱 층을 거친 뒤에 나오는 2D 출력인 Feature Map 생성 -> Pooling으로 해상도 축소<br>
Flatten으로 1차원 벡터화 -> Fully Connected Layer이 만들어짐-> Activation Function 활성화 함수 적용해 근사 -> Loss 계산이 가능해짐 -> Optimizer로 파라미터 업데이트 -> Epoch 반복 + Regularization 적용**

| 용어                              | 설명                                                                                                              |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Convolution (합성곱)**          | 입력 이미지에 커널(필터)을 슬라이딩하면서 내적 연산을 수행해 특징 맵(feature map)을 생성                           |
| **Kernel = Filter (커널 = 필터)** | 3×3, 5×5 같은 작은 크기의 가중치 행렬로, 각 국소 영역과 곱해져 입력에서 특정 패턴(엣지, 텍스처 등)을 추출            |
| **Stride (스트라이드)**           | 커널을 이동시킬 때 한 번에 이동하는 픽셀 수. (Stride=1: 한 픽셀씩, Stride=2: 두 픽셀씩 → 출력 맵 크기 변화)       |
| **Padding (패딩)**               | 입력 주변에 0(또는 다른 값)을 추가해 출력 크기를 조절하거나 경계 정보 손실을 방지 (‘same’: 입력과 같게, ‘valid’: 순수 합성곱) |
| **Activation Function**          | 합성곱·완전연결 층의 선형 출력에 비선형성을 부여. (대표: **ReLU**, Sigmoid, Tanh 등)                                |
| **Pooling (풀링)**               | 특징 맵 크기 축소 및 위치 변동 강인성 부여<br>- **Max Pooling**: 영역 내 최대값<br>- **Average Pooling**: 영역 내 평균값 |
| **Flatten**                      | 다차원 feature map을 1차원 벡터로 변환                                                                            |
| **Fully Connected Layer**        | 평탄화된 벡터를 입력으로 받아 최종 클래스 점수나 회귀값을 예측                                                     |
| **Epoch (에폭)**                 | 전체 학습 데이터를 한 번 모두 사용해 파라미터를 업데이트한 횟수                                                      |
| **Batch (배치)**                 | 한 번에 신경망에 입력으로 넣어 학습시키는 샘플 묶음. 배치 크기에 따라 학습 안정성과 속도 변화                         |
| **Loss Function**                | 모델 예측값과 실제값 간 차이를 수치화<br>– 분류: Cross-Entropy<br>– 회귀: MSE                                        |
| **Optimizer**                    | 손실 함수를 최소화하도록 파라미터를 업데이트하는 알고리즘<br>ex) SGD, Momentum, RMSprop, Adam 등                     |
| **Regularization (정규화)**       | 모델 복잡도 제어로 과적합 방지<br>– L1/L2 페널티: 가중치 크기 제어<br>– Dropout: 학습 시 뉴런 일부 무작위 비활성화    |
