# PCA로 적용한 n개의 차원을 사용한 SVM

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# data_mean = data.frame[['mean radius', 'mean texture', 'mean perimeter', 
#                         'mean area', 'mean smoothness', 'mean compactness', 'target']]
# sns.pairplot(data_mean, hue='target')
# plt.show()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
print(df.head())

# # 유방암 데이터의 입력 부분에 대한 히스토그램
# plt.figure(figsize=(10, 6))
# sns.pairplot(df, hue='target', diag_kind='kde', markers=["o", "s"], palette="Set2")
# plt.show()

X_df = df.iloc[:, :-1]  # feature만 골라오기 (target 제외)
print(X_df.mean(axis=0))
print(X_df.var(axis=0))

# 피처 스케일링: 학습 데이터
scalerX = StandardScaler()
scalerX.fit(data.data)
X_std = scalerX.transform(data.data)
# print(X_train_std)

pca = PCA()
pca.fit(X_std)

print(f'explained_variance_ :\n {pca.explained_variance_}')
print(f'explained_variance_ratio :\n {pca.explained_variance_ratio_}')

# 누적 분산 비율 계산
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print(f'Cumulative Variance Ratio:\n{cumulative_variance}')

# 95% 이상의 분산을 설명하는 주성분 개수 선택
n_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Number of components explaining 95% variance: {n_components}')

# Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.grid()
plt.show()

# plt.plot(pca.explained_variance_, marker='o')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Variance Ratio')
# plt.grid()
# plt.show()

Z = pca.transform(X_std)
Z_df = pd.DataFrame(data=Z, columns=[f'PC{i+1}' for i in range(Z.shape[1])])
print(Z_df.head())
Z_df['target'] = data['target']

# 주성분 히스토그램을 한 화면에 표시
n_cols = 3  # 한 행에 표시할 그래프 수
n_rows = (n_components + n_cols - 1) // n_cols  # 필요한 행의 수 계산

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))  # 서브플롯 생성
axes = axes.flatten()  # 2D 배열을 1D로 변환

for i, column in enumerate(Z_df.columns[:n_components]):  # 선택된 주성분 개수만큼 반복
    sns.histplot(data=Z_df, x=column, hue='target', bins=20, kde=True, palette={0: 'red', 1: 'green'}, alpha=0.7, ax=axes[i])
    axes[i].set_title(f'Histogram of {column} by Target (95% Variance)')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].grid()

# 남은 빈 플롯 숨기기
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()  # 레이아웃 조정
plt.show()

# plt.figure(figsize=(8, 8))
# ax = sns.pairplot(df, hue='target')
# plt.grid(True)
# plt.show()

# 학습용과 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(Z, data.target, test_size=0.3, random_state=42, stratify=data.target)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# 정확도를 저장할 리스트
accuracies = []

# 주성분 개수를 1부터 30까지 반복
for n_components in range(1, 31):
    # PCA 변환 (n_components 개수만큼 주성분 사용)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_std)
    
    # 학습용과 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(Z, data.target, test_size=0.3, random_state=42, stratify=data.target)
    
    # SVM 모델 학습
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    
    # 테스트 데이터 정확도 계산
    accuracy = clf.score(X_test, y_test)
    accuracies.append((n_components, accuracy))  # 주성분 개수와 정확도 저장

    print(f'Components: {n_components}, Accuracy: {accuracy:.4f}')

# 결과 시각화
components, scores = zip(*accuracies)
plt.figure(figsize=(10, 6))
plt.plot(components, scores, marker='o')
plt.title('Accuracy vs Number of Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(y_pred)

# # 혼동 행렬 및 정확도 출력
# cf = confusion_matrix(y_test, y_pred)
# print(f'Confusion Matrix:\n{cf}')
# print(f'Accuracy: {clf.score(X_test, y_test)}')