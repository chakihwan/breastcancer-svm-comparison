# 유방암 데이터 SVM 및 PCA 성능 비교

이 프로젝트는 Scikit-learn의 유방암 진단 데이터셋(Wisconsin Breast Cancer dataset)을 사용하여, 서포트 벡터 머신(SVM) 모델의 성능을 두 가지 조건에서 비교 분석합니다.

1.  원본 데이터(특성 30개)를 모두 사용한 SVM의 성능
2.  주성분 분석(PCA)으로 차원을 축소한 데이터를 사용한 SVM의 성능

## 1. 프로젝트 목표

유방암 데이터의 원본 특성(30개)을 모두 사용한 SVM 모델과, PCA를 적용하여 n개의 주성분만 사용한 SVM 모델의 정확도(Accuracy)를 비교합니다. 이를 통해 차원 축소(PCA)가 모델의 성능에 미치는 영향을 확인하고, 최소한의 주성분으로 원본과 유사한 성능을 낼 수 있는 지점을 탐색합니다.

## 2. 사용된 기술

* **Python 3**
* **Scikit-learn**:
    * `load_breast_cancer`: 데이터셋 로드
    * `train_test_split`: 데이터 분리
    * `StandardScaler`: 데이터 스케일링
    * `svm.SVC`: 서포트 벡터 머신
    * `PCA`: 주성분 분석
    * `accuracy_score`, `confusion_matrix`: 모델 성능 평가
* **Matplotlib / Seaborn**: 데이터 시각화
* **Pandas**: 데이터 전처리
<img width="382" height="586" alt="image" src="https://github.com/user-attachments/assets/6a55925c-0095-4b07-b3ee-196dc81b1724" />
*[데이터 셋트 개요]
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/8e79598a-bb98-4aaf-a896-f3fbc8104659" />
*[원본 데이터의 히스토그램]
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/21e20c72-1324-42bf-a492-07a4bd71f0dd" />
*['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness' 피처간의 관계 시각화]


## 3. 실험 과정

### 실험 1: 원본 데이터를 사용한 SVM (`svm_breastcancer_basic.py`)

1.  유방암 데이터를 로드하고 EDA(Pairplot)를 통해 시각화합니다.
2.  데이터를 훈련(80%)/테스트(20%) 세트로 분리합니다. (`test_size=0.2`, `stratify=data.target`)
3.  `StandardScaler`를 사용하여 훈련/테스트 데이터를 스케일링합니다.
4.  `svm.SVC(kernel='linear')` 모델을 훈련 세트로 학습시킵니다.
5.  테스트 세트로 모델의 정확도와 Confusion Matrix(혼동 행렬)를 계산하고 시각화합니다.

### 실험 2: PCA 적용 데이터를 사용한 SVM (`svm_breastcancer_pca.py`)

1.  유방암 데이터를 로드하고 `StandardScaler`를 사용하여 스케일링합니다.
2.  데이터를 훈련(70%)/테스트(30%) 세트로 분리합니다. (`test_size=0.3`)
3.  주성분의 수(n_components)를 2개부터 30개까지 1씩 증가시키며 다음을 반복합니다:
    a.  `PCA(n_components=i)` 객체를 생성하고 훈련 데이터에 `fit_transform`을 적용합니다.
    b.  테스트 데이터에는 `transform`을 적용합니다.
    c.  PCA로 변환된 데이터로 `LinearSVC` 모델을 학습시킵니다.
    d.  변환된 테스트 데이터로 정확도를 측정하고, "Components: [i], Accuracy: [score]" 형식으로 출력합니다.

---

## 4. 분석 결과 및 결론

### 실험 결과 요약

* **실험 1 (원본, 30개 특성, test_size=0.2)**:
    * `svm_breastcancer_basic.py` 실행 결과, 테스트 정확도는 **약 97.37%**로 확인되었습니다.
* **실험 2 (PCA, n개 특성, test_size=0.3)**:
    * `svm_breastcancer_pca.py` 실행 결과, 주성분 개수에 따라 정확도가 변동했습니다.
    * 주성분이 11개일 때 **약 98.25%**의 정확도를 기록했습니다. (이는 `test_size=0.3` 기준)
<img width="783" height="534" alt="image" src="https://github.com/user-attachments/assets/bbf48c28-a64c-4287-96ce-139bded523e8" />
<img width="273" height="513" alt="image" src="https://github.com/user-attachments/assets/89ad7159-b9b9-4e82-9911-77ffb1905f82" />

---



### ⭐️ 성능 비교 (Confusion Matrix) 

> `svm_breastcancer_basic.py`로 생성된 원본 이미지와, **새로 생성하신 PCA(n=11) Confusion Matrix 이미지**를 아래와 같이 나란히 배치하세요.

| 원본 데이터 (특성 30개, test_size=0.2) | PCA 적용 (주성분 11개, test_size=0.3) |
| :---: | :---: |
| `svm_breastcancer_basic.py` 실행 결과 | `svm_breastcancer_pca.py` (n=11) 실행 결과 |
| <img width="500" height="400" alt="Figure_11" src="https://github.com/user-attachments/assets/d854bbdf-b54f-4d94-a323-9aa2b59b6f43" /> | <img width="500" height="400" alt="Figure_12" src="https://github.com/user-attachments/assets/c0e2cb5b-e77e-4c5e-8777-ff8d2263f11b" /> |
| **정확도: 97.37%** (111/114) | **정확도: 98.25%** (168/171) |


---

### 결론 및 해석

* **분석 결과:**
    * 원본 데이터를 사용한 모델은 약 **97.37%**의 정확도를 보였습니다. (Test 20% 기준)
    * PCA를 적용한 모델은 주성분 11개일 때 약 **98.25%**의 정확도를 보였습니다. (Test 30% 기준)
* **주의:** 두 실험(`basic`과 `pca`)의 `test_size`가 (20% vs 30%) 다르기 때문에 정확도를 직접 비교하기에는 무리가 있습니다.
* **PCA의 의의 (실험 2 기준):** `svm_breastcancer_pca.py`의 결과만 놓고 보더라도, 데이터의 특성을 30개에서 11개로 **약 63%가량 줄였음에도 불구**하고, 여전히 **98.25%**라는 매우 높은 분류 성능을 달성할 수 있었습니다.
* **최종 결론**: 이는 PCA가 유방암 데이터의 분류에 유의미한 정보는 유지하면서 불필요한 노이즈나 중복 정보를 효과적으로 제거했음을 의미합니다. 모델의 학습 속도를 높이고 과적합 위험을 낮추는 차원 축소의 이점을 고려할 때, PCA(n=11)를 사용하는 것은 매우 합리적이고 효율적인 선택입니다.

## 5. 파일 구성

* `svm_breastcancer_basic.py`: 원본 데이터(30개 특성, test_size=0.2)로 SVM을 학습하고 평가/시각화하는 스크립트.
* `svm_breastcancer_pca.py`: PCA를 적용(n=2~30, test_size=0.3)하여 차원을 축소한 데이터로 SVM을 학습하고 평가하는 스크립트.


## 6. 실행 방법

1.  필요한 라이브러리를 설치합니다.
    ```bash
    pip install scikit-learn numpy matplotlib seaborn pandas
    ```

2.  원본 데이터 SVM 성능을 확인합니다. (Pairplot 및 Confusion Matrix 이미지가 생성됩니다)
    ```bash
    python svm_breastcancer_basic.py
    ```

3.  PCA 적용 SVM 성능을 확인합니다. (주성분별 정확도가 출력됩니다)
    ```bash
    python svm_breastcancer_pca.py
    ```
