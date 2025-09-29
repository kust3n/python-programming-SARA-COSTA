import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Labs/Laboration 3/unlabelled_data.csv", names=["x", "y"])

x = data["x"].values
y = data["y"].values

k = -1.5
m = 0.4

def line(x):
    return k * x + m

def classify_point(x, y):
    return y > line(x)

data["label"] = [int(classify_point(x[i], y[i])) for i in range(len(x))]
data.to_csv("Labs/Laboration 3/labelled_data.csv", index = False, header = False)

colors = ["green" if c == 1 else "blue" for c in data["label"]]
x_vals = np.array([data["x"].min(), data["x"].max()])
y_vals = line(x_vals)
plt.scatter(data["x"], data["y"], c = colors)
plt.plot(x_vals, y_vals, color="red", label=f"y = {k}x + {m}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Labelled Data")
plt.legend()
plt.show()


