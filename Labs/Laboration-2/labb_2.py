import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_points = "Labs/Laboration-2/datapoints.txt"
test_points = "Labs/Laboration-2/testpoints.txt"

def load_data_points(data_points):

    points = []

    with open(data_points, "r") as file:
        next(file)
        for line in file:

            line = line.strip().split(",")
            x, y, label = line
            
            points.append((float(x), float(y), int(label)))
    return points



def load_test_points(test_points):

    points = []

    with open(test_points, "r") as file:
        next(file)
        for line in file:
            
            line = line.strip()
            line = line.split("(")[1].strip(")").split(",")
            x, y = line
            points.append((float(x), float(y)))
    return points

train_points = load_test_points(test_points)
print(train_points[0])
