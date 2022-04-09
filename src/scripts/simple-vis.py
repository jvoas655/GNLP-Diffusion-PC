import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

if __name__ == "__main__":
    file = './lang_out.npy'
    data = np.load(file)

    gfile = './out.npy'
    gdata = np.load(gfile)

    for i in range(data.shape[0]):
        x = data[i, :, 0]
        y = data[i, :, 1]
        z = data[i, :, 2]

        fig = plt.figure(figsize=(16, 16))
        gax = fig.add_subplot(121, projection='3d')
        ax = fig.add_subplot(122, projection='3d')

        ax.scatter(x, y, z)
        # plt.show()


        gx = gdata[i, :, 0]
        gy = gdata[i, :, 1]
        gz = gdata[i, :, 2]

        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')

        gax.scatter(gx, gy, gz)
        plt.show()

        input('continue')