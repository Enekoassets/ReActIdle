from bvhTools import bvhIO, bvhSlicer, bvhMetrics, bvhVisualizerMpl
import numpy as np
import os

class speedClass:
    def __init__(self, speeds, id, videoId, seqId):
        self.speeds = speeds
        self.id = id
        self.videoId = videoId
        self.seqId = seqId

genuine_folder = "./genuine"
acted_folder = "./acted"
mode = "angular" # angular or linear
derivative = "speeds" # speeds, accelerations or jerks
avgGenuineSpeeds = []
avgActedSpeeds = []
id = 0
for file in os.listdir(genuine_folder):
    if file.endswith("bvh"):
        bvh = bvhIO.readBvh(os.path.join(genuine_folder + "/" + file))
        bvhSlices = bvhSlicer.getBvhSlices(bvh, [x for x in range(0, bvh.motion.numFrames-90, 90)], [x+89 for x in range(0, bvh.motion.numFrames-90, 90)])
        for seqId, bvhSlice in enumerate(bvhSlices):
            if(mode == "angular"):
                if derivative == "speeds":
                    avgGenuineSpeeds.append(speedClass(bvhMetrics.getAvgAngularSpeeds(bvhSlice, type="magnitude", mode="perJoint"), id, 1, seqId))
                elif derivative == "accelerations":
                    avgGenuineSpeeds.append(speedClass(bvhMetrics.getAvgAngularAccelerations(bvhSlice, type="magnitude", mode="perJoint"), id, 1, seqId))
                elif derivative == "jerks":
                    avgGenuineSpeeds.append(speedClass(bvhMetrics.getAvgAngularJerks(bvhSlice, type="magnitude", mode="perJoint"), id, 1, seqId))
            elif(mode == "linear"):
                if derivative == "speeds":
                    avgGenuineSpeeds.append(speedClass(bvhMetrics.getAvgSpeeds(bvhSlice, type="magnitude", mode="perJoint"), id, 1, seqId))
                elif derivative == "accelerations":
                    avgGenuineSpeeds.append(speedClass(bvhMetrics.getAvgAccelerations(bvhSlice, type="magnitude", mode="perJoint"), id, 1, seqId))
                elif derivative == "jerks":
                    avgGenuineSpeeds.append(speedClass(bvhMetrics.getAvgJerks(bvhSlice, type="magnitude", mode="perJoint"), id, 1, seqId))
        id+=1
id = 0
for file in os.listdir(acted_folder):
    if file.endswith("bvh"):
        bvh = bvhIO.readBvh(os.path.join(acted_folder + "/" + file))
        bvhSlices = bvhSlicer.getBvhSlices(bvh, [x for x in range(0, bvh.motion.numFrames-90, 90)], [x+89 for x in range(0, bvh.motion.numFrames-90, 90)])
        for seqId, bvhSlice in enumerate(bvhSlices):
            if(mode == "angular"):
                if derivative == "speeds":
                    avgActedSpeeds.append(speedClass(bvhMetrics.getAvgAngularSpeeds(bvhSlice, type="magnitude", mode="perJoint"), id, 0, seqId))
                elif derivative == "accelerations":
                    avgActedSpeeds.append(speedClass(bvhMetrics.getAvgAngularAccelerations(bvhSlice, type="magnitude", mode="perJoint"), id, 0, seqId))
                elif derivative == "jerks":
                    avgActedSpeeds.append(speedClass(bvhMetrics.getAvgAngularJerks(bvhSlice, type="magnitude", mode="perJoint"), id, 0, seqId))
            elif(mode == "linear"):
                if derivative == "speeds":
                    avgActedSpeeds.append(speedClass(bvhMetrics.getAvgSpeeds(bvhSlice, type="magnitude", mode="perJoint"), id, 0, seqId))
                elif derivative == "accelerations":
                    avgActedSpeeds.append(speedClass(bvhMetrics.getAvgAccelerations(bvhSlice, type="magnitude", mode="perJoint"), id, 0, seqId))
                elif derivative == "jerks":
                    avgActedSpeeds.append(speedClass(bvhMetrics.getAvgJerks(bvhSlice, type="magnitude", mode="perJoint"), id, 0, seqId))
        id+=1

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Convert to arrays
Xg = np.array([x.speeds for x in avgGenuineSpeeds])   # shape (269, 26)
Xa = np.array([x.speeds for x in avgActedSpeeds])     # shape (545, 26)

# Stack samples
X = np.vstack([Xg, Xa])           # shape (814, 26)

# Labels: 0 = genuine, 1 = acted
y = np.array([0]*len(Xg) + [1]*len(Xa))

class_colors = np.array(['red', 'green'])
class_labels = np.array(['Real', 'Acted'])
# 1) Standardize features
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

# 2) PCA
pca = PCA(n_components=2)
Xpca = pca.fit_transform(Xn)

plt.figure(figsize=(7,5))
for i, label in enumerate(class_labels):
    plt.scatter(
        Xpca[y == i, 0],
        Xpca[y == i, 1],
        color=class_colors[i],
        label=label,
        alpha=0.7
    )

plt.legend()
plt.title(f"PCA of average {mode} {derivative}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 3) t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    max_iter=1500
)
Xtsne = tsne.fit_transform(Xn)

plt.figure(figsize=(7,5))
for i, label in enumerate(class_labels):
    plt.scatter(
        Xtsne[y == i, 0],
        Xtsne[y == i, 1],
        color=class_colors[i],
        label=label,
        alpha=0.7
    )

plt.legend()
plt.title(f"t-SNE of average {mode} {derivative}")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
import umap

# 4) UMAP reducer
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)

X_umap = reducer.fit_transform(Xn)

plt.figure(figsize=(7,5))
for i, label in enumerate(class_labels):
    plt.scatter(
        X_umap[y == i, 0],
        X_umap[y == i, 1],
        color=class_colors[i],
        label=label,
        alpha=0.7
    )

plt.legend()
plt.title(f"UMAP embedding of average {mode} {derivative}")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()