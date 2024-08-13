import numpy as np
import matplotlib.pyplot as plt
from geomdl import fitting

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def fit_curve(curve_pts):
    curve = fitting.approximate_curve(curve_pts.tolist(), degree=2, ctrlpts_size=5)
    
    return curve


if __name__ == "__main__":

    for i in range(8):
        
        control_points = np.load("NameOfDumpDir/00000" + str(i) + "_controlpoints.npy")
        generatedDlos = np.load("NameOfDumpDir/00000" + str(i) + "_confident_bSplinePoints.npy")


        print(control_points.shape) # [maxNumObjects, 15]
        print(generatedDlos.shape) # [numconfident, 15]


        ax = plt.axes(projection='3d')
        set_axes_equal(ax)
        plt.cla()
        ax.set_box_aspect([1,1,1])

        for control in range(2): # Hard coded to the label size!
            # Reshape to 5 points
            ctrldlo = control_points[control, :].reshape((5,3))
            # Fit bspline
            ctrldloFitted = fit_curve(ctrldlo)
            # Take eval points
            evalpointsCtrlDlo = np.array(ctrldloFitted.evalpts)
            # Plot
            ax.plot(evalpointsCtrlDlo[:, 0], evalpointsCtrlDlo[:, 1], evalpointsCtrlDlo[:, 2], color="blue", linewidth=2.0, label=('label'+str(control)))
        
        for generated in range(generatedDlos.shape[0]):
            gendlo = generatedDlos[generated, :].reshape((5,3))
            gendloFitted = fit_curve(gendlo)
            evalpointsGenDlo = np.array(gendloFitted.evalpts)
            ax.plot(evalpointsGenDlo[:, 0], evalpointsGenDlo[:, 1], evalpointsGenDlo[:, 2], color="green", linewidth=2.0, label=('generated curve'+str(generated)))

        plt.axis('off')
        plt.legend()
        plt.draw()
        plt.show()