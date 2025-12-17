from bvhTools import bvhIO, bvhSlicer, bvhMetrics, bvhVisualizerMpl
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R

class speedClass:
    def __init__(self, speeds, id, videoId, seqId):
        self.speeds = speeds
        self.id = id
        self.videoId = videoId
        self.seqId = seqId

def load_mapping_xlsx(file_path, sheet_name="Sheet1"):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    mapping_bvh1 = set(df['Freemocap Rig'])  # Joints to keep in BVH1
    mapping_bvh2 = dict(zip(df['Mixamo Rig'], df['Freemocap Rig']))  # BVH2 to Freemocap Rig mapping
    return mapping_bvh1, mapping_bvh2

def filter_and_rename_bvh1(joint_names, rotations, mapping_bvh1):
    filtered_joint_names = []
    filtered_rotations = []
    
    for joint_idx, joint_name in enumerate(joint_names):
        if joint_name in mapping_bvh1:
            filtered_joint_names.append(joint_name)
            filtered_rotations.append(rotations[:, joint_idx])
    
    if filtered_rotations:
        filtered_rotations = np.stack(filtered_rotations, axis=1)
    else:
        filtered_rotations = np.empty((rotations.shape[0], 0, rotations.shape[2]))
    
    return filtered_joint_names, filtered_rotations

def filter_and_rename_bvh2(joint_names, rotations, mapping_bvh2):
    renamed_joint_names = []
    filtered_rotations = []
    
    for joint_idx, joint_name in enumerate(joint_names):
        if joint_name in mapping_bvh2:
            renamed_joint_names.append(mapping_bvh2[joint_name])
            filtered_rotations.append(rotations[:, joint_idx])
    
    if filtered_rotations:
        filtered_rotations = np.stack(filtered_rotations, axis=1)
    else:
        filtered_rotations = np.empty((rotations.shape[0], 0, rotations.shape[2]))
    
    return renamed_joint_names, filtered_rotations

def parse_bvh(file_path):
    joint_names = []
    positions = []
    rotations = []
    frame_time = None
    channel_info = []
    reading_motion = False
    total_channels = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("HIERARCHY"):
                continue
            
            if line.startswith("ROOT") or line.startswith("JOINT"):
                joint_names.append(line.split()[1])
            
            if line.startswith("CHANNELS"):
                tokens = line.split()
                channel_count = int(tokens[1])
                channels = tokens[2:]
                channel_info.append((channel_count, channels))
                total_channels += channel_count
            
            if line.startswith("MOTION"):
                reading_motion = True
                continue
            
            if reading_motion:
                if line.startswith("Frames:"):
                    continue
                
                if line.startswith("Frame Time:"):
                    frame_time = float(line.split()[-1])
                    continue
                
                # Parse frame motion data
                values = list(map(float, line.split()))
                if len(values) != total_channels:
                    raise ValueError(
                        f"Mismatch between total channels ({total_channels}) and motion data length ({len(values)})."
                    )
                
                frame_positions = []
                frame_rotations = []
                current_channel_start = 0
                
                for joint_idx, (channel_count, channels) in enumerate(channel_info):
                    joint_positions = [0, 0, 0]
                    joint_rotations = [0, 0, 0]
                    
                    for i, channel in enumerate(channels):
                        try:
                            value = values[current_channel_start + i]
                        except IndexError:
                            raise IndexError(
                                f"Index out of range while parsing joint {joint_names[joint_idx]} "
                                f"at channel {channel} (frame data length: {len(values)}, "
                                f"current_channel_start: {current_channel_start}, index: {current_channel_start + i})."
                            )
                        if channel.endswith("position"):
                            joint_positions[i % 3] = value
                        elif channel.endswith("rotation"):
                            joint_rotations[i % 3] = value
                    
                    current_channel_start += channel_count
                    
                    if any(c.endswith("position") for c in channels):
                        frame_positions.append(joint_positions)
                    else:
                        frame_positions.append(None)
                    
                    frame_rotations.append(joint_rotations)
                
                positions.append(frame_positions)
                rotations.append(frame_rotations)
    
    positions = np.array(positions, dtype=object)  
    rotations = np.array(rotations)
    
    return joint_names, positions, rotations, frame_time

def compute_angular_speeds_bvh_unwrapped(rotations, delta_time, order='xyz'):
    n_frames, n_joints, _ = rotations.shape
    speeds_per_joint = []
    
    # Unwrap the rotations to avoid discontinuities
    rotations_unwrapped = rotations.copy()
    for joint_idx in range(n_joints):
        for axis in range(3):  # Unwrap each rotation axis (X, Y, Z)
            rotations_unwrapped[:, joint_idx, axis] = np.unwrap(rotations[:, joint_idx, axis])
    
    for joint_idx in range(n_joints):
        joint_speeds = []
        
        for frame_idx in range(n_frames - 1):
            euler1 = rotations_unwrapped[frame_idx, joint_idx]
            euler2 = rotations_unwrapped[frame_idx + 1, joint_idx]
            
            # Convert to quaternions
            quat1 = R.from_euler(order, euler1, degrees=True).as_quat()
            quat2 = R.from_euler(order, euler2, degrees=True).as_quat()
            
            # Compute the dot product between the two quaternions
            dot_product = np.dot(quat1, quat2)
            
            # Ensure the dot product is within valid range for arccos
            dot_product = np.clip(dot_product, -1.0, 1.0)
            
            # Compute the angular difference in radians
            angular_difference = 2 * np.arccos(abs(dot_product))
            
            # Convert to degrees
            angular_difference_deg = np.degrees(angular_difference)
            if angular_difference_deg > 60:
                print(angular_difference_deg)
            
            # Calculate angular speed (deg/s)
            speed = angular_difference_deg / delta_time
            joint_speeds.append(speed)
        
        speeds_per_joint.append(np.array(joint_speeds))
    
    return speeds_per_joint

def compute_angular_accelerations(speeds, delta_time):
    accelerations_per_joint = []
    
    for joint_speeds in speeds:
        joint_accelerations = []
        
        for frame_idx in range(len(joint_speeds) - 1):
            acceleration = (joint_speeds[frame_idx + 1] - joint_speeds[frame_idx]) / delta_time
            joint_accelerations.append(acceleration)
        
        accelerations_per_joint.append(np.array(joint_accelerations))
    
    return accelerations_per_joint

def compute_angular_jerks(accelerations, delta_time):
    jerks_per_joint = []
    
    for joint_accelerations in accelerations:
        joint_jerks = []
        
        for frame_idx in range(len(joint_accelerations) - 1):
            jerk = (joint_accelerations[frame_idx + 1] - joint_accelerations[frame_idx]) / delta_time
            joint_jerks.append(jerk)
        
        jerks_per_joint.append(np.array(joint_jerks))
    
    return jerks_per_joint

genuine_folder = "./genuine/"
acted_folder = "./acted/"
mixamo_folder = "./mixamo/"
derivative = "speeds"
avgGenuineValues = []
avgActedValues = []
avgMixamoValues = []

mapping_bvh1, mapping_bvh2 = load_mapping_xlsx("mixamo2FreemocapMapping.xlsx")

for id, file_name in enumerate(os.listdir(genuine_folder)):
    if not file_name.endswith(".bvh"):
        continue
    file_path = os.path.join(genuine_folder, file_name)
    print(f"Processing file: {file_name} (Folder 0)")
    
    joint_names, _, rotations, frame_time = parse_bvh(file_path)
    
    joint_names, rotations = filter_and_rename_bvh1(joint_names, rotations, mapping_bvh1)
    fromFrames = [x for x in range(0, len(rotations)-90, 90)]
    toFrames = [x+99 for x in range(0, len(rotations)-90, 90)]
    for seqId, (fromFrame, toFrame) in enumerate(zip(fromFrames, toFrames)):
        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations[fromFrame:toFrame], frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        if(derivative == "speeds"):
            avgGenuineValues.append(speedClass([np.mean(joint_speeds) for joint_speeds in speeds_unwrapped], id, 0, seqId))
        elif(derivative == "accelerations"):
            avgGenuineValues.append(speedClass([np.mean(joint_accelerations) for joint_accelerations in accelerations], id, 0, seqId))
        elif(derivative == "jerks"):
            avgGenuineValues.append(speedClass([np.mean(joint_jerks) for joint_jerks in jerks], id, 0, seqId))

for id, file_name in enumerate(os.listdir(acted_folder)):
    if not file_name.endswith(".bvh"):
        continue
    file_path = os.path.join(acted_folder, file_name)
    print(f"Processing file: {file_name} (Folder 0)")
    
    joint_names, _, rotations, frame_time = parse_bvh(file_path)
    
    joint_names, rotations = filter_and_rename_bvh1(joint_names, rotations, mapping_bvh1)
    fromFrames = [x for x in range(0, len(rotations)-90, 90)]
    toFrames = [x+99 for x in range(0, len(rotations)-90, 90)]
    for seqId, (fromFrame, toFrame) in enumerate(zip(fromFrames, toFrames)):
        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations[fromFrame:toFrame], frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        if(derivative == "speeds"):
            avgActedValues.append(speedClass([np.mean(joint_speeds) for joint_speeds in speeds_unwrapped], id, 1, seqId))
        elif(derivative == "accelerations"):
            avgActedValues.append(speedClass([np.mean(joint_accelerations) for joint_accelerations in accelerations], id, 1, seqId))
        elif(derivative == "jerks"):
            avgActedValues.append(speedClass([np.mean(joint_jerks) for joint_jerks in jerks], id, 1, seqId))

for id, file_name in enumerate(os.listdir(mixamo_folder)):
    if not file_name.endswith(".bvh"):
        continue
    file_path = os.path.join(mixamo_folder, file_name)
    print(f"Processing file: {file_name} (Folder 0)")
    
    joint_names, _, rotations, frame_time = parse_bvh(file_path)
    
    joint_names, rotations = filter_and_rename_bvh2(joint_names, rotations, mapping_bvh2)
    fromFrames = [x for x in range(0, len(rotations)-90, 90)]
    toFrames = [x+99 for x in range(0, len(rotations)-90, 90)]
    for seqId, (fromFrame, toFrame) in enumerate(zip(fromFrames, toFrames)):
        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations[fromFrame:toFrame], frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        if(derivative == "speeds"):
            avgMixamoValues.append(speedClass([np.mean(joint_speeds) for joint_speeds in speeds_unwrapped], id, 2, seqId))
        elif(derivative == "accelerations"):
            avgMixamoValues.append(speedClass([np.mean(joint_accelerations) for joint_accelerations in accelerations], id, 2, seqId))
        elif(derivative == "jerks"):
            avgMixamoValues.append(speedClass([np.mean(joint_jerks) for joint_jerks in jerks], id, 2, seqId))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

# -----------------------------------------------------------
# Prepare arrays
# -----------------------------------------------------------

Xg = np.array([x.speeds for x in avgGenuineValues])   # (269,26)
Xa = np.array([x.speeds for x in avgActedValues])     # (545,26)
Xm = np.array([x.speeds for x in avgMixamoValues])     # (545,26)

print(Xg.shape, Xa.shape, Xm.shape)
# Stack features
X = np.vstack([Xg, Xa, Xm])                  # (814,26)
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

# Extract metadata in the same order
videoIds  = np.array([s.videoId for s in avgGenuineValues] +
                    [s.videoId for s in avgActedValues] + 
                    [s.videoId for s in avgMixamoValues])
ids       = np.array([s.id       for s in avgGenuineValues] +
                    [s.id       for s in avgActedValues] + 
                    [s.id       for s in avgMixamoValues])
y = np.array(
    [0]*len(Xg) +    # Genuine
    [1]*len(Xa) +    # Acted
    [2]*len(Xm)      # Mixamo
)
# Class names and colors
class_names = ["Real", "Acted", "Mixamo"]
class_colors = ["tab:red", "tab:green", "tab:blue"]

reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)

X_umap = reducer.fit_transform(Xn)

plt.figure(figsize=(7,5))
for cls, name, color in zip([0,1,2], class_names, class_colors):
    mask = (y == cls)
    plt.scatter(
        X_umap[mask,0],
        X_umap[mask,1],
        label=name,
        c=color,
        alpha=0.7,
        s=30
    )

plt.title(f"UMAP embedding of average angular {derivative}")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.show()

pca = PCA(n_components=2)
Xpca = pca.fit_transform(Xn)

plt.figure(figsize=(7,5))
for cls, name, color in zip([0,1,2], class_names, class_colors):
    mask = (y == cls)
    plt.scatter(
        Xpca[mask,0],
        Xpca[mask,1],
        label=name,
        c=color,
        alpha=0.7,
        s=30
    )

plt.title("PCA of Average Speeds")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    max_iter=1500,
    random_state=42
)

Xtsne = tsne.fit_transform(Xn)

plt.figure(figsize=(7,5))
for cls, name, color in zip([0,1,2], class_names, class_colors):
    mask = (y == cls)
    plt.scatter(
        Xtsne[mask,0],
        Xtsne[mask,1],
        label=name,
        c=color,
        alpha=0.7,
        s=30
    )

plt.title("t-SNE of Average Speeds")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.show()