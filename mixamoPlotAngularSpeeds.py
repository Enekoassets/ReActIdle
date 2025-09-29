import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

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


def compute_angular_speeds_bvh(rotations, delta_time, order='xyz'):
    n_frames, n_joints, _ = rotations.shape
    speeds_per_joint = []
    
    for joint_idx in range(n_joints):
        joint_speeds = []
        
        for frame_idx in range(n_frames - 1):
            euler1 = rotations[frame_idx, joint_idx]
            euler2 = rotations[frame_idx + 1, joint_idx]
            
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
            if(angular_difference_deg > 60):
                print(angular_difference_deg)
            # Calculate angular speed (deg/s)
            speed = angular_difference_deg / delta_time
            joint_speeds.append(speed)
        
        speeds_per_joint.append(np.array(joint_speeds))
    
    return speeds_per_joint

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

def plot_angular_speeds(joint_names, speeds, frame_time):
    n_frames = len(speeds[0])
    time = np.arange(n_frames) * frame_time

    plt.figure(figsize=(15, 10))
    
    for i, joint_speeds in enumerate(speeds):
        plt.plot(time[:], joint_speeds, label=joint_names[i], linewidth=2)

    plt.title("Angular Speeds of Joints Over Time", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Angular Speed (deg/s)", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def load_mapping_xlsx(file_path, sheet_name="Sheet1"):
    """
    Load the joint name mapping from an XLSX file.
    Args:
        file_path (str): Path to the XLSX file.
        sheet_name (str): Name of the sheet containing the mapping.
    Returns:
        tuple: Two dictionaries: mapping_bvh1 and mapping_bvh2.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    mapping_bvh1 = set(df['Freemocap Rig'])  # Joints to keep in BVH1
    mapping_bvh2 = dict(zip(df['Mixamo Rig'], df['Freemocap Rig']))  # BVH2 to Freemocap Rig mapping
    return mapping_bvh1, mapping_bvh2

def filter_and_rename_bvh1(joint_names, rotations, mapping_bvh1):
    """
    Filter joints in BVH1 based on the mapping.
    Args:
        joint_names (list): Original joint names in BVH1.
        rotations (np.ndarray): Rotations (frames x joints x 3).
        mapping_bvh1 (set): Set of joint names to retain.
    Returns:
        tuple: Filtered joint names and rotations.
    """
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
    """
    Filter and rename joints in BVH2 based on the mapping.
    Args:
        joint_names (list): Original joint names in BVH2.
        rotations (np.ndarray): Rotations (frames x joints x 3).
        mapping_bvh2 (dict): Mapping from BVH2 joint names to BVH1 joint names.
    Returns:
        tuple: Renamed joint names and filtered rotations.
    """
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


def check_and_report_high_speeds(joint_names, rotations, speeds, threshold=1000):
    for joint_idx, joint_speeds in enumerate(speeds):
        for frame_idx, speed in enumerate(joint_speeds):
            if speed > threshold:
                print(
                    f"High angular speed detected! "
                    f"Joint: {joint_names[joint_idx]}, "
                    f"Frame: {frame_idx + 1}, "
                    f"Speed: {speed:.2f} degrees/sec, "
                    f"Rotation: {rotations[frame_idx, joint_idx]}, "
                    f"Rotation next: {rotations[frame_idx + 1, joint_idx]}, "
                    f"Rotation last: {rotations[frame_idx - 1, joint_idx]}"
                )

def process_bvh_folders(folder_path_0, folder_path_1, folder_path_2):

    bvh_files_0 = [f for f in os.listdir(folder_path_0) if f.endswith('.bvh')]
    bvh_files_1 = [f for f in os.listdir(folder_path_1) if f.endswith('.bvh')]
    bvh_files_2 = [f for f in os.listdir(folder_path_2) if f.endswith('.bvh')]
    print(len(bvh_files_1))
    plt.figure(figsize=(12, 8))
    
    avg_speeds_list_0 = []
    avg_accelerations_list_0 = []
    avg_jerks_list_0 = []
    avg_speeds_list_1 = []
    avg_accelerations_list_1 = []
    avg_jerks_list_1 = []
    avg_speeds_list_2 = []
    avg_accelerations_list_2 = []
    avg_jerks_list_2 = []
    
    mapping_bvh1, mapping_bvh2 = load_mapping_xlsx("mixamo2FreemocapMapping.xlsx")

    for i, file_name in enumerate(bvh_files_0):
        file_path = os.path.join(folder_path_0, file_name)
        print(f"Processing file: {file_name} (Folder 0)")
        
        joint_names, _, rotations, frame_time = parse_bvh(file_path)
        
        joint_names, rotations = filter_and_rename_bvh1(joint_names, rotations, mapping_bvh1)
        
        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations, frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        
        avg_speeds_list_0.append([np.mean(joint_speeds) for joint_speeds in speeds_unwrapped])
        avg_accelerations_list_0.append([np.mean(joint_accelerations) for joint_accelerations in accelerations])
        avg_jerks_list_0.append([np.mean(joint_jerks) for joint_jerks in jerks])

    for i, file_name in enumerate(bvh_files_1):
        file_path = os.path.join(folder_path_1, file_name)
        print(f"Processing file: {file_name} (Folder 1)")
        
        joint_names, _, rotations, frame_time = parse_bvh(file_path)
        
        joint_names, rotations = filter_and_rename_bvh1(joint_names, rotations, mapping_bvh1)
        
        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations, frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        
        avg_speeds_list_1.append([np.mean(joint_speeds) for joint_speeds in speeds_unwrapped])
        avg_accelerations_list_1.append([np.mean(joint_accelerations) for joint_accelerations in accelerations])
        avg_jerks_list_1.append([np.mean(joint_jerks) for joint_jerks in jerks])

    for i, file_name in enumerate(bvh_files_2):
        file_path = os.path.join(folder_path_2, file_name)
        print(f"Processing file: {file_name} (Folder 2)")
        
        joint_names, _, rotations, frame_time = parse_bvh(file_path)

        joint_names, rotations = filter_and_rename_bvh2(joint_names, rotations, mapping_bvh2)

        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations, frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        
        avg_speeds_list_2.append([np.mean(joint_speeds) for joint_speeds in speeds_unwrapped])
        avg_accelerations_list_2.append([np.mean(joint_accelerations) for joint_accelerations in accelerations])
        avg_jerks_list_2.append([np.mean(joint_jerks) for joint_jerks in jerks])

    ####################
    ###### SPEEDS ######
    ####################

    avg_speeds_folder_0 = np.mean(avg_speeds_list_0, axis=0)
    plt.plot(joint_names, avg_speeds_folder_0, label='Real Idle Average', color='red', linewidth=3)

    avg_speeds_folder_1 = np.mean(avg_speeds_list_1, axis=0)
    plt.plot(joint_names, avg_speeds_folder_1, label='Acted Idle Average', color='green', linewidth=3)

    avg_speeds_folder_2 = np.mean(avg_speeds_list_2, axis=0)
    plt.plot(joint_names, avg_speeds_folder_2, label='Mixamo Idle Average', color='blue', linewidth=3)

    std_speeds_folder_0 = np.std(avg_speeds_list_0, axis=0)
    plt.fill_between(
        joint_names,
        avg_speeds_folder_0 - std_speeds_folder_0,
        avg_speeds_folder_0 + std_speeds_folder_0,
        color='red',
        alpha=0.1,
        label='Real Idle Std Dev'
    )

    std_speeds_folder_1 = np.std(avg_speeds_list_1, axis=0)
    plt.fill_between(
        joint_names,
        avg_speeds_folder_1 - std_speeds_folder_1,
        avg_speeds_folder_1 + std_speeds_folder_1,
        color='green',
        alpha=0.1,
        label='Acted Idle Std Dev'
    )

    std_speeds_folder_2 = np.std(avg_speeds_list_2, axis=0)
    plt.fill_between(
        joint_names,
        avg_speeds_folder_2 - std_speeds_folder_2,
        avg_speeds_folder_2 + std_speeds_folder_2,
        color='blue',
        alpha=0.1,
        label='Mixamo Idle Std Dev'
    )
    
    plt.title("Average Angular Speeds from recorded and Mixamo idle animations")
    plt.xlabel("Joint Names")
    plt.ylabel("Speed (deg/s)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()

    ###################
    ## ACCELERATIONS ##
    ###################
    avg_accelerations_folder_0 = np.mean(avg_accelerations_list_0, axis=0)
    plt.plot(joint_names, avg_accelerations_folder_0, label='Real Idle Average', color='red', linewidth=3)

    avg_accelerations_folder_1 = np.mean(avg_accelerations_list_1, axis=0)
    plt.plot(joint_names, avg_accelerations_folder_1, label='Acted Idle Average', color='green', linewidth=3)

    avg_accelerations_folder_2 = np.mean(avg_accelerations_list_2, axis=0)
    plt.plot(joint_names, avg_accelerations_folder_2, label='Mixamo Idle Average', color='blue', linewidth=3)

    std_accelerations_folder_0 = np.std(avg_accelerations_list_0, axis=0)
    std_accelerations_folder_1 = np.std(avg_accelerations_list_1, axis=0)
    std_accelerations_folder_2 = np.std(avg_accelerations_list_2, axis=0)
    plt.fill_between(
        joint_names,
        avg_accelerations_folder_0 - std_accelerations_folder_0,
        avg_accelerations_folder_0 + std_accelerations_folder_0,
        color='red',
        alpha=0.1,
        label='Real Idle Std Dev'
    )
    plt.fill_between(
        joint_names,
        avg_accelerations_folder_1 - std_accelerations_folder_1,
        avg_accelerations_folder_1 + std_accelerations_folder_1,
        color='green',
        alpha=0.1,
        label='Acted Idle Std Dev'
    )
    plt.fill_between(
        joint_names,
        avg_accelerations_folder_2 - std_accelerations_folder_2,
        avg_accelerations_folder_2 + std_accelerations_folder_2,
        color='blue',
        alpha=0.1,
        label='Mixamo Idle Std Dev'
    )

    plt.title("Average Angular Accelerations from recorded and Mixamo idle animations")
    plt.xlabel("Joint Names")
    plt.ylabel("Acceleration (deg/s^2)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    ###################
    ###### JERKS ######
    ###################
    avg_jerks_folder_0 = np.mean(avg_jerks_list_0, axis=0)
    plt.plot(joint_names, avg_jerks_folder_0, label='Real Idle Average', color='red', linewidth=3)

    avg_jerks_folder_1 = np.mean(avg_jerks_list_1, axis=0)
    plt.plot(joint_names, avg_jerks_folder_1, label='Acted Idle Average', color='green', linewidth=3)

    avg_jerks_folder_2 = np.mean(avg_jerks_list_2, axis=0)
    plt.plot(joint_names, avg_jerks_folder_2, label='Mixamo Idle Average', color='blue', linewidth=3)

    std_jerks_folder_0 = np.std(avg_jerks_list_0, axis=0)
    std_jerks_folder_1 = np.std(avg_jerks_list_1, axis=0)
    std_jerks_folder_2 = np.std(avg_jerks_list_2, axis=0)

    plt.fill_between(
        joint_names,
        avg_jerks_folder_0 - std_jerks_folder_0,
        avg_jerks_folder_0 + std_jerks_folder_0,
        color='red',
        alpha=0.1,
        label='Real Idle Std Dev'
    )
    plt.fill_between(
        joint_names,
        avg_jerks_folder_1 - std_jerks_folder_1,
        avg_jerks_folder_1 + std_jerks_folder_1,
        color='green',
        alpha=0.1,
        label='Acted Idle Std Dev'
    )
    plt.fill_between(
        joint_names,
        avg_jerks_folder_2 - std_jerks_folder_2,
        avg_jerks_folder_2 + std_jerks_folder_2,
        color='blue',
        alpha=0.1,
        label='Mixamo Idle Std Dev'
    )

    plt.title("Average Angular Jerks from recorded and Mixamo idle animations")
    plt.xlabel("Joint Names")
    plt.ylabel("jerk (deg/s^3)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()

folder_path_0 = "./genuine/"
folder_path_1 = "./acted"
folder_path_2 = "./mixamo/" 
process_bvh_folders(folder_path_0, folder_path_1, folder_path_2)