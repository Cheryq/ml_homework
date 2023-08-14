import numpy as np
import torch
def PCA(X,k):
    X = X - np.mean(X, axis=0)
    X_cov = np.cov(X.T, ddof=1)

    eigenvalues,eigenvectors = np.linalg.eig(X_cov) #计算协方差矩阵的特征值和特征向量
    top_k_eigenvectors = eigenvectors[:, :k]
    X = np.dot(X, top_k_eigenvectors)
    return X
