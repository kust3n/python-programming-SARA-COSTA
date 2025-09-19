import matplotlib.pyplot as plt
import numpy as np

data_points = "Labs/Laboration-2/datapoints.txt"
test_points = "Labs/Laboration-2/testpoints.txt"

def load_data_points(data_points):

    points = []

    with open(data_points, "r") as file:
        next(file)
        for line in file:

            row = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            x, y, label = row
            
            points.append((float(x), float(y), int(label)))
    return points



def load_test_points(test_points):

    points = []

    with open(test_points, "r") as file:
        next(file)
        for line in file:
            
            line = line.strip()
            row = line.split("(")[1].strip(")").split(",")
            x, y = row
            points.append((float(x), float(y)))
    return points

train_points = load_test_points(test_points)
print(train_points)
