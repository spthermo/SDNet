import numpy as np
import matplotlib.pyplot as plt

def save_anatomy_factors(y, folder, iteration):
    x = y[0]
    for j in range(y.shape[0]):
        if j > 0:
            x = np.concatenate((x, y[j]), axis=1)

    plt.imsave(folder + '/anatomy_factors_' + str(iteration) + '.jpg', x, cmap='gray')