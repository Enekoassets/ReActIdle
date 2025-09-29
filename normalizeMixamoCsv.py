import pandas as pd
import glob
import os
import csv
import numpy as np
import os
from bvh import Bvh

def parse_offset(bvh, joint_name):
    """Extract the OFFSET for a joint as a numpy array."""
    offset = bvh.joint_offset(joint_name)  # This returns a list of strings
    return np.array([float(v) for v in offset])

def calculate_skeleton_size(bvh, exclude_joints=None):
    """Calculate the size of the skeleton, including bone ends based on hierarchical positions."""
    if exclude_joints is None:
        exclude_joints = set()
    else:
        exclude_joints = set(exclude_joints)

    def has_end_site(bvh, joint_name):
        """Check if a joint node has an 'End Site' child."""
        childCount = 0
        for child in bvh.joint_direct_children(joint_name):
            if child.name not in exclude_joints:
                childCount += 1
        # return len(bvh.joint_direct_children(joint_name)) == 0
        return childCount == 0

    def get_end_site_offset(joint_node):
        for child in joint_node.children:
            for ichild in child.children:
                if("OFFSET" in str(ichild)):
                    return np.array([float(str(ichild).split(" ")[1]), float(str(ichild).split(" ")[2]), float(str(ichild).split(" ")[3])])
        return np.array([0, 0, 0])
    # def get_end_site_offset(joint_node):
    #     """Get the offset for the 'End Site' child of a joint node."""
    #     for child in joint_node.children:
    #         if child.name == "End Site":
    #             return np.array([float(v) for v in child['OFFSET']])
    #     return np.array([0, 0, 0])  # Default if no End Site is found

    def traverse_joint(joint_name, parent_position):
        global_positions = []

        # Calculate the current joint's global position
        joint_offset = parse_offset(bvh, joint_name)
        current_position = parent_position + joint_offset

        # Add the current joint's position
        global_positions.append(current_position)

        # Add the bone end position if it has an 'End Site'
        joint_node = bvh.get_joint(joint_name)
        if has_end_site(bvh, joint_name):
            end_site_offset = get_end_site_offset(joint_node)
            print(joint_name)
            print("start: ", current_position)
            bone_end_position = current_position + end_site_offset
            print("end: ", bone_end_position)
            global_positions.append(bone_end_position)

        # Process child joints recursively
        for child_node in bvh.joint_direct_children(joint_name):
            child_name = child_node.name
            if child_name in exclude_joints:
                continue
            global_positions.extend(traverse_joint(child_name, current_position))

        return global_positions

    # Start from the root joint
    root_joint_name = bvh.get_joints_names()[0]
    root_position = np.array([0, 0, 0])  # Root starts at the origin
    global_positions = traverse_joint(root_joint_name, root_position)

    # Calculate size metrics
    global_positions = np.array(global_positions)
    min_coords = np.min(global_positions, axis=0)
    max_coords = np.max(global_positions, axis=0)
    dimensions = max_coords - min_coords
    height = dimensions[1]  # Assuming Y-axis is up
    width = dimensions[0]
    depth = dimensions[2]
    return height, width, depth

def getScale(bvhFileName):
    bvhFilePath = os.path.join("./mixamo/", bvhFileName + ".bvh")
    # Load the BVH file
    with open(bvhFilePath, "r") as file:
        bvh_data = Bvh(file.read())

    return calculate_skeleton_size(bvh_data, exclude_joints=["mixamorig:Head"])

def load_data(csv_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)
    coordinates = data.iloc[:, 0:].values  # Remaining columns as coordinates
    header = data.columns  # Preserve the original header
    return  coordinates, header

# Set the folder path containing the CSV files
folder_path = './mixamo/csv/*.csv'  # Replace with your folder path
output_folder = './mixamo/csv/normalized/'
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

# Iterate over all CSV files in the specified folder
for csv_file in glob.glob(folder_path):
    print(f"Processing file: {csv_file}")

    # Load data
    coordinates, header = load_data(csv_file)

    # Scale the data depending on the size of the skeleton
    height, width, depth = getScale(os.path.basename(csv_file).strip("_pos.csv"))
    print("height:", height, "width:", width, "depth:", depth)

    # Scale the coordinates
    scaled_coordinates = []
    for coord in coordinates:
        scaled_coord = coord.copy()  # Create a copy to avoid modifying the original
        for i in range(0, coord.shape[0]-2, 3):
            scaled_coord[i] /= height
            scaled_coord[i+1] /= height
            scaled_coord[i+2] /= height
        scaled_coordinates.append(scaled_coord)

    # Write scaled coordinates to a new CSV file
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(csv_file))[0] + "_normalized.csv")
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the original header
        writer.writerow(header)
        # Write data rows
        for scaled_coord in scaled_coordinates:
            writer.writerow(list(scaled_coord))

    print(f"Scaled data written to: {output_file}")