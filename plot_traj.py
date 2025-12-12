import numpy as np
import matplotlib.pyplot as plt

poses = np.load("trajectory.npy")   # (N, 3)
x, y, z = poses[:, 0], poses[:, 1], poses[:, 2]

plt.figure()
plt.plot(x, z, "-")
plt.xlabel("x")
plt.ylabel("z")
plt.axis("equal")
plt.title("Camera trajectory (top-down x–z view)")
plt.show()
