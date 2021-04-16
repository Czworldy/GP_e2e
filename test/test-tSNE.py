import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets

digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target
print(X.shape, y.shape, y[0], type(y[0]))

X = np.concatenate([
	np.random.randn(100, 128)+1,
	np.random.randn(100, 128)-1,
	np.random.randn(100, 128)-0.5,
	np.random.randn(100, 128)+0.5,
])

y = np.array([1]*100 + [2]*100 + [3]*100 + [4]*100)
print(X.shape, y.shape, y[0])
# n_samples, n_features = X.shape

'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca')
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()