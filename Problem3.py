import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
CONTOUR_LEVELS = np.geomspace(0.0001, 250, 100)
MARKER_TOTAL = 4
NOISE_SIGMA = 0.3
SIGMAX = 0.9
SIGMAY = 0.9
SIG_ARRAY = np.array([[pow(SIGMAX,2),0],[0,pow(SIGMAY,2)]])
LEVELS = np.geomspace(1, 300, 100)

random.seed(3)
if __name__ == '__main__':
    # Initialize the true position of the vehicle
    r = random.random()
    theta = random.uniform(0, 2)
    x = r * np.cos(theta * np.pi)
    y = r * np.sin(theta * np.pi)
    xy_t = np.array((x, y))

    # Generate the locations of the landmarks
    markers = np.zeros((MARKER_TOTAL, 2))
    for i in range(MARKER_TOTAL):
        theta = (2 * np.pi) / MARKER_TOTAL * i
        x = np.cos(theta)
        y = np.sin(theta)
        markers[i,:] = np.array((x,y))
        distance_from_true = []
        for marker in markers[:i+1]:
            distance_from_true.append(np.linalg.norm(xy_t-marker))

        # Need to generate a coordinate field. For this example, 512x512
        grid_orig = np.meshgrid(np.linspace(-2, 2, 512), np.linspace(-2, 2, 512))
        grid = np.expand_dims(np.transpose(grid_orig, axes=(1,2,0)), axis=len(np.shape(grid_orig))-1)

        # Equation 28 in report, first portion
        # If the grid was not resized, np.swapaxes() would not be used and a transpose would work instead
        pO = np.matmul(np.matmul(grid, np.linalg.inv(SIG_ARRAY)), np.swapaxes(grid, 2, 3))[:,0,0,0]

        # Need to calculate the second portion (sum) of equation 28 in report
        range = []
        for marker_index, distance in enumerate(distance_from_true):
            grid_distance = np.linalg.norm(grid - markers[marker_index][None, None, None, :], axis=3)
            range.append(np.squeeze(pow(distance-grid_distance,2)/pow(NOISE_SIGMA,2)))
        range = sum(range)
        param_estimate_grid = pO+range

        # Plot Findings

        fig0 = plt.figure(0, figsize=[10, 10])
        ax = fig0.add_subplot(1, 1, 1)
        circ = plt.Circle((0, 0), radius=1, edgecolor='r', facecolor='None')
        ax.scatter(xy_t[0], xy_t[1], 100, marker='+')
        color_list = ['g', 'b', 'y', 'm', 'k', 'c','g', 'b', 'y', 'm', 'k', 'c','g', 'b', 'y', 'm', 'k', 'c','g', 'b', 'y', 'm', 'k', 'c']
        for marker_index, distance in enumerate(distance_from_true):
            xy = markers[marker_index]
            ax.scatter(xy[0], xy[1], 100, marker = 'o', color = color_list[marker_index])
            ax.add_patch(plt.Circle((xy[0],xy[1]), radius=distance, edgecolor=color_list[marker_index], facecolor='None'))
        ax.set_xlim((-2,2))
        ax.set_ylim((-2,2))
        plt.contour(grid_orig[0], grid_orig[1], param_estimate_grid, levels = CONTOUR_LEVELS)

        ax.add_patch(circ)
        plt.show()
        print(f'{r=}')
        print(f'{theta=} * Pi')


