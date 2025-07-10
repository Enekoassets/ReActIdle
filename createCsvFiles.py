from bvhTools import bvhIO
import os

os.makedirs("./acted/csv", exist_ok = True)
os.makedirs("./genuine/csv", exist_ok = True)

for file in os.listdir("./acted/"):
    if file.endswith("bvh"):
        bvhIO.writePositionsToCsv(bvhIO.readBvh("./acted/" + file), "./acted/csv/" + file.strip(".bvh") + "_pos.csv")

for file in os.listdir("./genuine/"):
    if file.endswith("bvh"):
        bvhIO.writePositionsToCsv(bvhIO.readBvh("./genuine/" + file), "./genuine/csv/" + file.strip(".bvh") + "_pos.csv")