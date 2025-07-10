import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

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
    time_differences = [0.04167 for x in range(len(coordinates)-1)]
    speeds = []

    num_points = coordinates.shape[1] // 3
    
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
    time_differences = [0.04167 for x in range(len(coordinates)-1)]

    accelerations = []

    for speeds_for_point in speeds:
        # Calculate accelerations (change in speed / time)
        if len(speeds_for_point) > 1:
            accels_for_point = np.diff(speeds_for_point) / time_differences[:len(speeds_for_point) - 1]
            accelerations.append(accels_for_point)
        else:
            accelerations.append(np.array([]))
    return accelerations

def compute_jerks(accelerations):
    time_differences = [0.04167 for x in range(len(coordinates)-1)]

    jerks = []

    for accels_for_point in accelerations:
        # Calculate jerks (change in acceleration / time)
        if len(accels_for_point) > 1:
            jerks_for_point = np.diff(accels_for_point) / time_differences[:len(accels_for_point) - 1]
            jerks.append(jerks_for_point)
        else:
            jerks.append(np.array([]))

    return jerks

folder_path_1 = './genuine/csv/*.csv'
folder_path_2 = './acted/csv/*.csv'

plt.figure(figsize=(10, 6))

average_speeds_list_1 = []
average_accelerations_list_1 = []
average_jerks_list_1 = []
file_names_1 = []
joint_names_1 = []

for csv_file in glob.glob(folder_path_1):
    coordinates, joint_names_1 = load_data(csv_file)

    speeds = compute_speeds(coordinates)
    accelerations = compute_accelerations(speeds)
    jerks = compute_jerks(accelerations)
    average_speeds = [np.mean(speeds_for_point) for speeds_for_point in speeds]
    average_accelerations = [np.mean(accels_for_point) for accels_for_point in accelerations]
    average_jerks = [np.mean(jerks_for_point) for jerks_for_point in jerks]

    average_speeds_list_1.append(average_speeds)
    average_accelerations_list_1.append(average_accelerations)
    average_jerks_list_1.append(average_jerks)
    file_names_1.append(csv_file.split("/")[-1])

average_speeds_list_2 = []
average_accelerations_list_2 = []
average_jerks_list_2 = []
file_names_2 = []
joint_names_2 = []
for csv_file in glob.glob(folder_path_2):
    coordinates, joint_names_2 = load_data(csv_file)

    speeds = compute_speeds(coordinates)
    accelerations = compute_accelerations(speeds)
    jerks = compute_jerks(accelerations)
    average_speeds = [np.mean(speeds_for_point) for speeds_for_point in speeds]
    average_accelerations = [np.mean(accels_for_point) for accels_for_point in accelerations]
    average_jerks = [np.mean(jerks_for_point) for jerks_for_point in jerks]

    average_speeds_list_2.append(average_speeds)
    average_accelerations_list_2.append(average_accelerations)
    average_jerks_list_2.append(average_jerks)
    file_names_2.append(csv_file.split("/")[-1])

############
## SPEEDS ##
############

avg_speeds_folder_1 = np.mean(average_speeds_list_1, axis=0)
avg_speeds_folder_2 = np.mean(average_speeds_list_2, axis=0)

std_dev_1 = np.std(average_speeds_list_1, axis=0)
std_dev_2 = np.std(average_speeds_list_2, axis=0)

plt.plot(avg_speeds_folder_1, label='Real Idle Average', color='red', linewidth=3)

plt.plot(avg_speeds_folder_2, label='Acted Idle Average', color='green', linewidth=3)

plt.fill_between(
    joint_names_1,
    avg_speeds_folder_1 - std_dev_1,
    avg_speeds_folder_1 + std_dev_1,
    color='red',
    alpha=0.1,
    label='Real Idle Std Dev'
)

plt.fill_between(
    joint_names_1,
    avg_speeds_folder_2 - std_dev_2,
    avg_speeds_folder_2 + std_dev_2,
    color='green',
    alpha=0.1,
    label='Acted Idle Std Dev'
)

plt.title('Average Speeds from Real and Acted idle animations')
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

avg_accelerations_folder_1 = np.mean(average_accelerations_list_1, axis=0)
avg_accelerations_folder_2 = np.mean(average_accelerations_list_2, axis=0)

std_dev_1 = np.std(average_accelerations_list_1, axis=0)
std_dev_2 = np.std(average_accelerations_list_2, axis=0)

plt.plot(avg_accelerations_folder_1, label='Real Idle Average', color='red', linewidth=3)

plt.plot(avg_accelerations_folder_2, label='Acted Idle Average', color='green', linewidth=3)

plt.fill_between(
    joint_names_1,
    avg_accelerations_folder_1 - std_dev_1,
    avg_accelerations_folder_1 + std_dev_1,
    color='red',
    alpha=0.1,
    label='Real Idle Std Dev'
)

plt.fill_between(
    joint_names_1,
    avg_accelerations_folder_2 - std_dev_2,
    avg_accelerations_folder_2 + std_dev_2,
    color='green',
    alpha=0.1,
    label='Acted Idle Std Dev'
)

plt.title('Average Accelerations from Real and Acted idle animations')
plt.xlabel('Joint Names')
plt.ylabel('Acceleration (m/s^2)')
plt.xticks(ticks=np.arange(len(average_speeds_list_1[0])), labels=joint_names_1, rotation = 90)
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()

###########
## JERKS ##
###########

avg_jerks_folder_1 = np.mean(average_jerks_list_1, axis=0)
avg_jerks_folder_2 = np.mean(average_jerks_list_2, axis=0)

std_dev_1 = np.std(average_jerks_list_1, axis=0)
std_dev_2 = np.std(average_jerks_list_2, axis=0)

plt.plot(avg_jerks_folder_1, label='Real Idle Average', color='red', linewidth=3)

plt.plot(avg_jerks_folder_2, label='Acted Idle Average', color='green', linewidth=3)

plt.fill_between(
    joint_names_1,
    avg_jerks_folder_1 - std_dev_1,
    avg_jerks_folder_1 + std_dev_1,
    color='red',
    alpha=0.1,
    label='Real Idle Std Dev'
)

plt.fill_between(
    joint_names_1,
    avg_jerks_folder_2 - std_dev_2,
    avg_jerks_folder_2 + std_dev_2,
    color='green',
    alpha=0.1,
    label='Acted Idle Std Dev'
)

plt.title('Average Jerks from Real and Acted idle animations')
plt.xlabel('Joint Names')
plt.ylabel('Jerk (m/s^3)')
plt.xticks(ticks=np.arange(len(average_speeds_list_1[0])), labels=joint_names_1, rotation = 90)
plt.grid()
plt.legend()
plt.tight_layout()

plt.show()

print("Processing complete.")