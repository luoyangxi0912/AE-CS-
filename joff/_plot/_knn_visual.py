import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def cluster_and_visualize(X, y_onehot, n_clusters=None, random_seed=42):
    """
    对数据进行聚类并用t-SNE可视化

    参数:
    X: 输入数据，形状为(n_samples, n_features)
    y_onehot: 独热编码标签，形状为(n_samples, n_classes)
    n_clusters: 聚类数量，如果为None则使用标签类别数
    random_seed: 随机种子
    """
    if type(X) == list: X = np.concatenate(X, axis = 0)
    if type(y_onehot) == list: y_onehot = np.concatenate(y_onehot, axis=0)
    print(X.shape, y_onehot.shape)
    # 将独热编码转换为原始标签
    if y_onehot.ndim > 1:
        y_true = np.argmax(y_onehot, axis=1)
    else:
        y_true = y_onehot
    true_labels = np.unique(y_true)

    # 如果没有指定聚类数量，使用真实标签的类别数
    if n_clusters is None:
        n_clusters = len(true_labels)

    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    y_pred = kmeans.fit_predict(X)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=random_seed)
    X_embedded = tsne.fit_transform(X)

    # 创建可视化图表
    plt.figure(figsize=(15, 6))

    # 子图1: 真实标签的可视化
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                           c=y_true, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter1, label='True Labels')
    plt.title('t-SNE Visualization with True Labels')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # 子图2: 聚类结果的可视化
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                           c=y_pred, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter2, label='Cluster Labels')
    plt.title(f't-SNE Visualization with K-means Clustering (k={n_clusters})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.tight_layout()
    plt.show()

    # 打印一些基本信息
    print(f"数据形状: {X.shape}")
    print(f"真实标签类别数: {len(true_labels)}")
    print(f"聚类数量: {n_clusters}")

    return {
        'true_labels': y_true,
        'cluster_labels': y_pred,
        'tsne_embedding': X_embedded,
        'kmeans_model': kmeans
    }

def example():
    # 示例数据 - 在实际使用中替换为你的数据
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import OneHotEncoder

    # 生成示例数据
    X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)

    # 将标签转换为独热编码
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y_true.reshape(-1, 1))

    # 调用函数
    result = cluster_and_visualize(X, y_onehot)

# 使用示例
if __name__ == "__main__":
    example()