import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

if __name__ == "__main__":
    file = './lang_out.npy'
    data = np.load(file)

    x = data[0, :, 0]
    y = data[0, :, 1]
    z = data[0, :, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)
    plt.show()


    file = './out.npy'
    data = np.load(file)

    x = data[0, :, 0]
    y = data[0, :, 1]
    z = data[0, :, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)
    plt.show()