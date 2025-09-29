import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    
    coordinates = data.iloc[:, 0:].values
    column_names = data.columns.values
    
    cleaned_column_names = [name.rsplit('_', 1)[0] for name in column_names]

    unique_joint_names = []
    seen = set()
    
    for name in cleaned_column_names:
        if name not in seen:
            unique_joint_names.append(name)
            seen.add(name)
    
    return coordinates, unique_joint_names

def compute_speeds(coordinates):
    time_differences = [0.03333 for x in range(len(coordinates)-1)]

    speeds = []

    # Loop through each point (each column of coordinates)
    num_points = coordinates.shape[1] // 3  # Number of points (x, y, z sets)
    
    for point in range(num_points):
        # Extract x, y, z coordinates for this point across all frames
        x_coords = coordinates[:, point * 3]
        y_coords = coordinates[:, point * 3 + 1]
        z_coords = coordinates[:, point * 3 + 2]

        # Calculate the distances between consecutive frames
        distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2 + np.diff(z_coords)**2)
        
        # Calculate speeds for this point (distance / time)
        if len(time_differences) > 0:
            speeds_for_point = distances / time_differences
            speeds.append(speeds_for_point)
        else:
            speeds.append(np.array([]))
    return speeds

def compute_accelerations(speeds):
    time_differences = [0.03333 for x in range(len(coordinates)-1)]

    accelerations = []

    for speeds_for_point in speeds:
        # Calculate accelerations (change in speed / time)
        if len(speeds_for_point) > 1:
            accels_for_point = np.diff(speeds_for_point) / time_differences[:len(speeds_for_point) - 1]
            accelerations.append(accels_for_point)
        else:
            accelerations.append(np.array([]))  # No accelerations if not enough data

    return accelerations

def compute_jerks(accelerations):
    time_differences = [0.03333 for x in range(len(coordinates)-1)]

    jerks = []

    for accels_for_point in accelerations:
        # Calculate jerks (change in acceleration / time)
        if len(accels_for_point) > 1:
            jerks_for_point = np.diff(accels_for_point) / time_differences[:len(accels_for_point) - 1]
            jerks.append(jerks_for_point)
        else:
            jerks.append(np.array([]))  # No jerks if not enough data

    return jerks

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
    # TODO: Check if this is correct
    for joint_idx, joint_name in enumerate(joint_names):
        joint_idx = joint_idx * 3
        if joint_name in mapping_bvh1:
            filtered_joint_names.append(joint_name)
            filtered_rotations.append(rotations[:, joint_idx])
            filtered_rotations.append(rotations[:, joint_idx+1])
            filtered_rotations.append(rotations[:, joint_idx+2])
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
        joint_idx = joint_idx * 3
        if joint_name in mapping_bvh2:
            renamed_joint_names.append(mapping_bvh2[joint_name])
            filtered_rotations.append(rotations[:, joint_idx])
            filtered_rotations.append(rotations[:, joint_idx+1])
            filtered_rotations.append(rotations[:, joint_idx+2])
    
    if filtered_rotations:
        filtered_rotations = np.stack(filtered_rotations, axis=1)
    else:
        filtered_rotations = np.empty((rotations.shape[0], 0, rotations.shape[2]))

    return renamed_joint_names, filtered_rotations

folder_path_0 = './genuine/csv/*.csv'
folder_path_1 = './acted/csv/*.csv'
folder_path_2 = './mixamo/csv/normalized/*.csv'
mapping_bvh1, mapping_bvh2 = load_mapping_xlsx("mixamo2FreemocapPointMapping.xlsx")

plt.figure(figsize=(10, 6))

average_speeds_list_0 = []
average_accelerations_list_0 = []
average_jerks_list_0 = []
file_names_0 = []
joint_names_0 = []

for csv_file in glob.glob(folder_path_0):
    coordinates, joint_names_0 = load_data(csv_file)
    
    joint_names_0, coordinates = filter_and_rename_bvh1(joint_names_0, coordinates, mapping_bvh1)

    speeds = compute_speeds(coordinates)
    accelerations = compute_accelerations(speeds)
    jerks = compute_jerks(accelerations)
    average_speeds = [np.mean(speeds_for_point) for speeds_for_point in speeds]
    average_accelerations = [np.mean(accels_for_point) for accels_for_point in accelerations]
    average_jerks = [np.mean(jerks_for_point) for jerks_for_point in jerks]

    average_speeds_list_0.append(average_speeds)
    average_accelerations_list_0.append(average_accelerations)
    average_jerks_list_0.append(average_jerks)
    file_names_0.append(csv_file.split("/")[-1])  # Store filename only

average_speeds_list_1 = []
average_accelerations_list_1 = []
average_jerks_list_1 = []
file_names_1 = []
joint_names_1 = []

for csv_file in glob.glob(folder_path_1):
    coordinates, joint_names_1 = load_data(csv_file)

    joint_names_1, coordinates = filter_and_rename_bvh1(joint_names_1, coordinates, mapping_bvh1)

    speeds = compute_speeds(coordinates)
    accelerations = compute_accelerations(speeds)
    jerks = compute_jerks(accelerations)
    average_speeds = [np.mean(speeds_for_point) for speeds_for_point in speeds]
    average_accelerations = [np.mean(accels_for_point) for accels_for_point in accelerations]
    average_jerks = [np.mean(jerks_for_point) for jerks_for_point in jerks]

    average_speeds_list_1.append(average_speeds)
    average_accelerations_list_1.append(average_accelerations)
    average_jerks_list_1.append(average_jerks)
    file_names_1.append(csv_file.split("/")[-1])  # Store filename only

average_speeds_list_2 = []
average_accelerations_list_2 = []
average_jerks_list_2 = []
file_names_2 = []
joint_names_2 = []
for csv_file in glob.glob(folder_path_2):
    coordinates, joint_names_2 = load_data(csv_file)

    joint_names_2, coordinates = filter_and_rename_bvh2(joint_names_2, coordinates, mapping_bvh2)

    speeds = compute_speeds(coordinates)
    accelerations = compute_accelerations(speeds)
    jerks = compute_jerks(accelerations)
    average_speeds = [np.mean(speeds_for_point) for speeds_for_point in speeds]
    average_accelerations = [np.mean(accels_for_point) for accels_for_point in accelerations]
    average_jerks = [np.mean(jerks_for_point) for jerks_for_point in jerks]

    average_speeds_list_2.append(average_speeds)
    average_accelerations_list_2.append(average_accelerations)
    average_jerks_list_2.append(average_jerks)
    file_names_2.append(csv_file.split("/")[-1])  # Store filename only

reordered_average_speeds_2 = []
reordered_average_accelerations_2 = []
reordered_average_jerks_2 = []
reordered_joint_names_2 = []

average_speeds_list_2 = list(zip(*average_speeds_list_2))
average_accelerations_list_2 = list(zip(*average_accelerations_list_2))
average_jerks_list_2 = list(zip(*average_jerks_list_2))

for joint_name in joint_names_1:
    if joint_name in joint_names_2:
        index = joint_names_2.index(joint_name)
        reordered_average_speeds_2.append(average_speeds_list_2[index])
        reordered_average_accelerations_2.append(average_accelerations_list_2[index])
        reordered_average_jerks_2.append(average_jerks_list_2[index])
        reordered_joint_names_2.append(joint_names_2[index])
    else:
        reordered_average_speeds_2.append([])
        reordered_average_accelerations_2.append([])
        reordered_average_jerks_2.append([])
        reordered_joint_names_2.append("joint_name not found in folder 2: " + joint_name)
print(reordered_joint_names_2)
reordered_average_speeds_2 = list(zip(*reordered_average_speeds_2))
reordered_average_accelerations_2 = list(zip(*reordered_average_accelerations_2))
reordered_average_jerks_2 = list(zip(*reordered_average_jerks_2))

average_speeds_list_2 = list(zip(*average_speeds_list_2))
average_accelerations_list_2 = list(zip(*average_accelerations_list_2))
average_jerks_list_2 = list(zip(*average_jerks_list_2))

# in joint_names_1 change the "shoulder.R" to "spine.002"
joint_names_1 = [name.replace("shoulder.R", "spine.002") for name in joint_names_1]
# save the index of spine.002
spine_002_index = joint_names_1.index("spine.002")
# transpose of average_speeds_list_1
average_speeds_list_0 = list(zip(*average_speeds_list_0))
average_speeds_list_1 = list(zip(*average_speeds_list_1))
reordered_average_speeds_2 = list(zip(*reordered_average_speeds_2))
average_accelerations_list_0 = list(zip(*average_accelerations_list_0))
average_accelerations_list_1 = list(zip(*average_accelerations_list_1))
reordered_average_accelerations_2 = list(zip(*reordered_average_accelerations_2))
average_jerks_list_0 = list(zip(*average_jerks_list_0))
average_jerks_list_1 = list(zip(*average_jerks_list_1))
reordered_average_jerks_2 = list(zip(*reordered_average_jerks_2))
# put the columns of spine.002 on the second position
joint_names_1.insert(1, joint_names_1.pop(joint_names_1.index("spine.002")))
average_speeds_list_0.insert(1, average_speeds_list_0.pop(spine_002_index))
average_speeds_list_1.insert(1, average_speeds_list_1.pop(spine_002_index))
reordered_average_speeds_2.insert(1, reordered_average_speeds_2.pop(spine_002_index))
average_accelerations_list_0.insert(1, average_accelerations_list_0.pop(spine_002_index))
average_accelerations_list_1.insert(1, average_accelerations_list_1.pop(spine_002_index))
reordered_average_accelerations_2.insert(1, reordered_average_accelerations_2.pop(spine_002_index))
average_jerks_list_0.insert(1, average_jerks_list_0.pop(spine_002_index))
average_jerks_list_1.insert(1, average_jerks_list_1.pop(spine_002_index))
reordered_average_jerks_2.insert(1, reordered_average_jerks_2.pop(spine_002_index))
# transpose again
average_speeds_list_0 = list(zip(*average_speeds_list_0))
average_speeds_list_1 = list(zip(*average_speeds_list_1))
reordered_average_speeds_2 = list(zip(*reordered_average_speeds_2))
average_accelerations_list_0 = list(zip(*average_accelerations_list_0))
average_accelerations_list_1 = list(zip(*average_accelerations_list_1))
reordered_average_accelerations_2 = list(zip(*reordered_average_accelerations_2))
average_jerks_list_0 = list(zip(*average_jerks_list_0))
average_jerks_list_1 = list(zip(*average_jerks_list_1))
reordered_average_jerks_2 = list(zip(*reordered_average_jerks_2))

############
## SPEEDS ##
############
avg_speeds_folder_0 = np.mean(average_speeds_list_0, axis=0)
avg_speeds_folder_1 = np.mean(average_speeds_list_1, axis=0)
avg_speeds_folder_2 = np.mean(reordered_average_speeds_2, axis=0)

std_dev_0 = np.std(average_speeds_list_0, axis=0)
std_dev_1 = np.std(average_speeds_list_1, axis=0)
std_dev_2 = np.std(reordered_average_speeds_2, axis=0)

plt.plot(avg_speeds_folder_0, label='Real Idle Average', color='red', linewidth=3)
plt.plot(avg_speeds_folder_1, label='Acted Idle Average', color='green', linewidth=3)
plt.plot(avg_speeds_folder_2, label='Mixamo Idle Average', color='blue', linewidth=3)

std_speeds_folder_0 = np.std(avg_speeds_folder_0, axis=0)
plt.fill_between(
    joint_names_0,
    avg_speeds_folder_0 - std_dev_0,
    avg_speeds_folder_0 + std_dev_0,
    color='red',
    alpha=0.1,
    label='Real Idle Std Dev'
)

std_speeds_folder_1 = np.std(avg_speeds_folder_1, axis=0)
plt.fill_between(
    joint_names_0,
    avg_speeds_folder_1 - std_dev_1,
    avg_speeds_folder_1 + std_dev_1,
    color='green',
    alpha=0.1,
    label='Acted Idle Std Dev'
)

std_speeds_folder_2 = np.std(avg_speeds_folder_2, axis=0)
plt.fill_between(
    joint_names_0,
    avg_speeds_folder_2 - std_dev_2,
    avg_speeds_folder_2 + std_dev_2,
    color='blue',
    alpha=0.1,
    label='Mixamo Idle Std Dev'
)

plt.title('Average Speeds from recorded and Mixamo idle animations')
plt.xlabel('Joint Names')
plt.ylabel('Speed (m/s)')
plt.xticks(ticks=np.arange(len(average_speeds_list_1[0])), labels=joint_names_1, rotation = 90)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

###################
## ACCELERATIONS ##
###################

avg_accelerations_folder_0 = np.mean(average_accelerations_list_0, axis=0)
avg_accelerations_folder_1 = np.mean(average_accelerations_list_1, axis=0)
avg_accelerations_folder_2 = np.mean(reordered_average_accelerations_2, axis=0)

std_dev_0 = np.std(average_accelerations_list_0, axis=0)
std_dev_1 = np.std(average_accelerations_list_1, axis=0)
std_dev_2 = np.std(reordered_average_accelerations_2, axis=0)

plt.plot(avg_accelerations_folder_0, label='Real Idle Average', color='red', linewidth=3)
plt.plot(avg_accelerations_folder_1, label='Acted Idle Average', color='green', linewidth=3)
plt.plot(avg_accelerations_folder_2, label='Mixamo Idle Average', color='blue', linewidth=3)

plt.fill_between(
    joint_names_0,
    avg_accelerations_folder_0 - std_dev_0,
    avg_accelerations_folder_0 + std_dev_0,
    color='red',
    alpha=0.1,
    label='Real Idle Std Dev'
)

plt.fill_between(
    joint_names_0,
    avg_accelerations_folder_1 - std_dev_1,
    avg_accelerations_folder_1 + std_dev_1,
    color='green',
    alpha=0.1,
    label='Acted Idle Std Dev'
)

plt.fill_between(
    joint_names_0,
    avg_accelerations_folder_2 - std_dev_2,
    avg_accelerations_folder_2 + std_dev_2,
    color='blue',
    alpha=0.1,
    label='Mixamo Idle Std Dev'
)

plt.title('Average Accelerations from recorded and Mixamo idle animations')
plt.xlabel('Joint Names')
plt.ylabel('Acceleration (m/s^2)')
plt.xticks(ticks=np.arange(len(average_speeds_list_1[0])), labels=joint_names_1, rotation = 90)
plt.grid()
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

###########
## JERKS ##
###########

avg_jerks_folder_0 = np.mean(average_jerks_list_0, axis=0)
avg_jerks_folder_1 = np.mean(average_jerks_list_1, axis=0)
avg_jerks_folder_2 = np.mean(reordered_average_jerks_2, axis=0)

std_dev_0 = np.std(average_jerks_list_0, axis=0)
std_dev_1 = np.std(average_jerks_list_1, axis=0)
std_dev_2 = np.std(reordered_average_jerks_2, axis=0)

plt.plot(avg_jerks_folder_0, label='Real Idle Average', color='red', linewidth=3)
plt.plot(avg_jerks_folder_1, label='Acted Idle Average', color='green', linewidth=3)
plt.plot(avg_jerks_folder_2, label='Mixamo Idle Average', color='blue', linewidth=3)

plt.fill_between(
    joint_names_0,
    avg_jerks_folder_0 - std_dev_0,
    avg_jerks_folder_0 + std_dev_0,
    color='red',
    alpha=0.1,
    label='Real Idle Std Dev'
)

plt.fill_between(
    joint_names_0,
    avg_jerks_folder_1 - std_dev_1,
    avg_jerks_folder_1 + std_dev_1,
    color='green',
    alpha=0.1,
    label='Acted Idle Std Dev'
)

plt.fill_between(
    joint_names_0,
    avg_jerks_folder_2 - std_dev_2,
    avg_jerks_folder_2 + std_dev_2,
    color='blue',
    alpha=0.1,
    label='Mixamo Idle Std Dev'
)

plt.title('Average Jerks from recorded and Mixamo idle animations')
plt.xlabel('Joint Names')
plt.ylabel('Jerk (m/s^3)')
plt.xticks(ticks=np.arange(len(average_speeds_list_1[0])), labels=joint_names_1, rotation = 90)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print("Processing complete.")
