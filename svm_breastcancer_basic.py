# 원본 데이터 그대로 사용한 SVM 코드

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터 로드
data = load_breast_cancer(as_frame=True)
# 데이터프레임 출력
print(data.frame)
# 입력 부분과 목표 값 출력
print(data.data)
print(data.target)
# print(type(data.data))
# print(data.data.shape)
# print(data.DESCR)

data_mean = data.frame[['mean radius', 'mean texture', 'mean perimeter', 
                        'mean area', 'mean smoothness', 'mean compactness', 'target']]
sns.pairplot(data_mean, hue='target')
plt.show()

# # 유방암 데이터의 입력 부분에 대한 히스토그램
# data.data.hist(figsize=(15, 10), bins=20)
# plt.suptitle('Histogram of Breast Cancer Features', fontsize=16)
# plt.show()

# 학습용과 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# 탐색적 데이터 분석
# 산포도 
# # 학습용 데이터
# plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train)
# # 테스트용 데이터
# plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, marker='x', s=100)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Scatter plot of training and test data')
# plt.show()

# 피처 스케일링: 학습 데이터
scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_std = scalerX.transform(X_train)
# print(X_train_std)

# 피처 스케일링: 테스트 데이터
X_test_std = scalerX.transform(X_test)
# print(X_test_std)

clf = svm.SVC(kernel='linear')
clf.fit(X_train_std, y_train)
y_pred = clf.predict(X_test_std)
print(y_pred)

# 혼동 행렬
cf = confusion_matrix(y_test, y_pred)
print(cf)

# 테스트 데이터에 대한 정확도
print(clf.score(X_test_std, y_test))

# 혼동 행렬 시각화
plt.figure(figsize=(6, 5))
sns.heatmap(cf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: Malignant(0)', 'Pred: Benign(1)'],
            yticklabels=['Actual: Malignant(0)', 'Actual: Benign(1)'])
plt.title('Confusion Matrix (SVM - Breast Cancer)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
