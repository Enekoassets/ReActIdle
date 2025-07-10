import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def parse_bvh(file_path):
    joint_names = []
    positions = []
    rotations = []
    frame_time = None
    channel_info = []  # Store channel type information for each joint
    reading_motion = False
    total_channels = 0  # Total number of channels in the file

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
                        frame_positions.append(None)  # No position channels for this joint
                    
                    frame_rotations.append(joint_rotations)
                
                positions.append(frame_positions)
                rotations.append(frame_rotations)
    
    # Convert positions and rotations to NumPy arrays where possible
    positions = np.array(positions, dtype=object)  # Object dtype for handling None
    rotations = np.array(rotations)  # Shape: (frames, joints, 3)
    
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
            # Calculate acceleration
            acceleration = (joint_speeds[frame_idx + 1] - joint_speeds[frame_idx]) / delta_time
            joint_accelerations.append(acceleration)
        
        accelerations_per_joint.append(np.array(joint_accelerations))
    
    return accelerations_per_joint

def compute_angular_jerks(accelerations, delta_time):
    jerks_per_joint = []
    
    for joint_accelerations in accelerations:
        joint_jerks = []
        
        for frame_idx in range(len(joint_accelerations) - 1):
            # Calculate jerk
            jerk = (joint_accelerations[frame_idx + 1] - joint_accelerations[frame_idx]) / delta_time
            joint_jerks.append(jerk)
        
        jerks_per_joint.append(np.array(joint_jerks))
    
    return jerks_per_joint

def plot_angular_speeds(joint_names, speeds, frame_time):
    n_frames = len(speeds[0])
    time = np.arange(n_frames) * frame_time  # Timeline based on frame time

    plt.figure(figsize=(15, 10))
    
    # Plot angular speeds for each joint
    for i, joint_speeds in enumerate(speeds):
        plt.plot(time[:], joint_speeds, label=joint_names[i], linewidth=2)

    # Customize plot
    plt.title("Angular Speeds of Joints Over Time", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Angular Speed (deg/s)", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Modify the compute_angular_speeds_bvh function to include the check
def check_and_report_high_speeds(joint_names, rotations, speeds, threshold=1000):
    for joint_idx, joint_speeds in enumerate(speeds):
        for frame_idx, speed in enumerate(joint_speeds):
            if speed > threshold:
                print(
                    f"High angular speed detected! "
                    f"Joint: {joint_names[joint_idx]}, "
                    f"Frame: {frame_idx + 1}, "  # Frames are 1-indexed
                    f"Speed: {speed:.2f} degrees/sec, "
                    f"Rotation: {rotations[frame_idx, joint_idx]}, "
                    f"Rotation next: {rotations[frame_idx + 1, joint_idx]}, "
                    f"Rotation last: {rotations[frame_idx - 1, joint_idx]}"
                )

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def process_bvh_folders(folder_path_1, folder_path_2):
    bvh_files_1 = [f for f in os.listdir(folder_path_1) if f.endswith('.bvh')]
    bvh_files_2 = [f for f in os.listdir(folder_path_2) if f.endswith('.bvh')]
    
    plt.figure(figsize=(12, 8))
    
    # Prepare lists to store the average speeds for each folder
    avg_speeds_list_1 = []
    avg_accelerations_list_1 = []
    avg_jerks_list_1 = []
    avg_speeds_list_2 = []
    avg_accelerations_list_2 = []
    avg_jerks_list_2 = []

    for i, file_name in enumerate(bvh_files_1):
        file_path = os.path.join(folder_path_1, file_name)
        print(f"Processing file: {file_name} (Folder 1 - Orange cmap)")
        
        # Parse BVH file
        joint_names, positions, rotations, frame_time = parse_bvh(file_path)
        # Compute angular speeds
        # speeds = compute_angular_speeds_bvh(rotations, frame_time)
        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations, frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        # Calculate average speeds for each joint
        avg_speeds = [np.mean(joint_speeds) for joint_speeds in speeds_unwrapped]
        avg_accelerations = [np.mean(joint_accelerations) for joint_accelerations in accelerations]
        avg_jerks = [np.mean(joint_jerks) for joint_jerks in jerks]
        avg_speeds_list_1.append(avg_speeds)
        avg_accelerations_list_1.append(avg_accelerations)
        avg_jerks_list_1.append(avg_jerks)
    
    for i, file_name in enumerate(bvh_files_2):
        file_path = os.path.join(folder_path_2, file_name)
        print(f"Processing file: {file_name} (Folder 2 - Blue cmap)")
        
        # Parse BVH file
        joint_names, positions, rotations, frame_time = parse_bvh(file_path)
        
        # Compute angular speeds
        # speeds = compute_angular_speeds_bvh(rotations, frame_time)
        speeds_unwrapped = compute_angular_speeds_bvh_unwrapped(rotations, frame_time)
        accelerations = compute_angular_accelerations(speeds_unwrapped, frame_time)
        jerks = compute_angular_jerks(accelerations, frame_time)
        # Calculate average speeds for each joint
        avg_speeds = [np.mean(joint_speeds) for joint_speeds in speeds_unwrapped]
        avg_accelerations = [np.mean(joint_accelerations) for joint_accelerations in accelerations]
        avg_jerks = [np.mean(joint_jerks) for joint_jerks in jerks]
        avg_speeds_list_2.append(avg_speeds)
        avg_accelerations_list_2.append(avg_accelerations)
        avg_jerks_list_2.append(avg_jerks)

    ####################
    ###### SPEEDS ######
    ####################

    # Compute and plot the average of all files in the first folder (red line)
    avg_speeds_folder_1 = np.mean(avg_speeds_list_1, axis=0)
    plt.plot(joint_names, avg_speeds_folder_1, label='Real Idle Average', color='red', linewidth=3)

    # Compute and plot the average of all files in the second folder (green line)
    avg_speeds_folder_2 = np.mean(avg_speeds_list_2, axis=0)
    plt.plot(joint_names, avg_speeds_folder_2, label='Acted Idle Average', color='green', linewidth=3)

    std_speeds_folder_1 = np.std(avg_speeds_list_1, axis=0)
    plt.fill_between(
        joint_names,
        avg_speeds_folder_1 - std_speeds_folder_1,
        avg_speeds_folder_1 + std_speeds_folder_1,
        color='red',
        alpha=0.1,  # Adjust opacity
        label='Real Idle Std Dev'
    )

    std_speeds_folder_2 = np.std(avg_speeds_list_2, axis=0)
    plt.fill_between(
        joint_names,
        avg_speeds_folder_2 - std_speeds_folder_2,
        avg_speeds_folder_2 + std_speeds_folder_2,
        color='green',
        alpha=0.1,  # Adjust opacity
        label='Acted Idle Std Dev'
    )

    # Customize the plot
    plt.title("Average Angular Speeds from real and acted idle animations")
    plt.xlabel("Joint Names")
    plt.ylabel("Speed (deg/s)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Add the legend
    # plt.legend(title="BVH Files", bbox_to_anchor=(1.05, 1), loc='upper left')  # Add legend outside the plot
    plt.legend()
    plt.grid(True)
    plt.show()

    ###################
    ## ACCELERATIONS ##
    ###################
    # Compute and plot the average of all files in the first folder (red line)
    print(np.array(avg_accelerations_list_1).shape)
    avg_accelerations_folder_1 = np.mean(avg_accelerations_list_1, axis=0)
    plt.plot(joint_names, avg_accelerations_folder_1, label='Real Idle Average', color='red', linewidth=3)

    # Compute and plot the average of all files in the second folder (green line)
    avg_accelerations_folder_2 = np.mean(avg_accelerations_list_2, axis=0)
    plt.plot(joint_names, avg_accelerations_folder_2, label='Acted Idle Average', color='green', linewidth=3)

    # Compute and plot the standard deviation of the average speeds for the first folder
    std_accelerations_folder_1 = np.std(avg_accelerations_list_1, axis=0)
    plt.fill_between(
        joint_names,
        avg_accelerations_folder_1 - std_accelerations_folder_1,
        avg_accelerations_folder_1 + std_accelerations_folder_1,
        color='red',
        alpha=0.1,  # Adjust opacity
        label='Real Idle Std Dev'
    )
    
    # # Compute and plot the standard deviation of the average speeds for the second folder
    std_accelerations_folder_2 = np.std(avg_accelerations_list_2, axis=0)
    plt.fill_between(
        joint_names,
        avg_accelerations_folder_2 - std_accelerations_folder_2,
        avg_accelerations_folder_2 + std_accelerations_folder_2,
        color='green',
        alpha=0.1,  # Adjust opacity
        label='Acted Idle Std Dev'
    )
    
    # Customize the plot
    plt.title("Average Angular Accelerations from real and acted idle animations")
    plt.xlabel("Joint Names")
    plt.ylabel("Acceleration (deg/s^2)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Add the legend
    plt.legend()
    plt.grid(True)
    plt.show()

    ###################
    ###### JERKS ######
    ###################

    # Compute and plot the average of all files in the first folder (red line)
    avg_jerks_folder_1 = np.mean(avg_jerks_list_1, axis=0)
    plt.plot(joint_names, avg_jerks_folder_1, label='Real Idle Average', color='red', linewidth=3)

    # Compute and plot the average of all files in the second folder (green line)
    avg_jerks_folder_2 = np.mean(avg_jerks_list_2, axis=0)
    plt.plot(joint_names, avg_jerks_folder_2, label='Acted Idle Average', color='green', linewidth=3)

    # Compute and plot the standard deviation of the average speeds for the first folder
    std_jerks_folder_1 = np.std(avg_jerks_list_1, axis=0)
    plt.plot(joint_names, std_jerks_folder_1 + avg_jerks_folder_1, label='Real Idle Std Dev', color='red', linestyle=':', linewidth=1)

    std_jerks_folder_1 = np.std(avg_jerks_list_1, axis=0)
    plt.plot(joint_names, -std_jerks_folder_1 + avg_jerks_folder_1, color='red', linestyle=':', linewidth=1)

    # Compute and plot the standard deviation of the average speeds for the second folder
    std_jerks_folder_2 = np.std(avg_jerks_list_2, axis=0)
    plt.plot(joint_names, std_jerks_folder_2 + avg_jerks_folder_2, label='Acted Idle Std Dev', color='green', linestyle=':', linewidth=1)

    std_jerks_folder_2 = np.std(avg_jerks_list_2, axis=0)
    plt.plot(joint_names, -std_jerks_folder_2 + avg_jerks_folder_2, color='green', linestyle=':', linewidth=1)

    # Customize the plot
    plt.title("Average Angular Jerks from real and acted idle animations")
    plt.xlabel("Joint Names")
    plt.ylabel("jerk (deg/s^3)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Add the legend
    plt.legend()
    plt.grid(True)
    plt.show()
    print("\n")

# Example usage
folder_path_1 = "./genuine/"
folder_path_2 = "./acted/"
process_bvh_folders(folder_path_1, folder_path_2)