import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# # 模拟数据 替换为自己的路径
# batch_size = 100
# X = np.random.rand(batch_size, 1, 100)  # 模拟一维数据，形状为 (100, 1, 100)
# y = np.random.randint(0, 2, size=(batch_size,))  # 模拟标签 (0 或 1)

# 数据预处理
X = X.squeeze(1)  # 将 (100, 1, 100) 转换为 (100, 100)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化数据

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 初始化SVM模型
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')  # 使用径向基核函数 (RBF)

# 训练模型
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
