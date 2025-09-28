import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Labs/Laboration 3/unlabelled_data.csv", names=["x", "y"])

plt.scatter(data["x"], data["y"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Unlabeled data:")

k = 1
m = 0

x_vals = data['x']
y_vals = k * x_vals + m

plt.scatter(data['x'], data['y'])
plt.plot(x_vals, y_vals, color='red', label=f'y = {k}x + {m}')
plt.legend()

plt.show()

