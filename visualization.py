''' Code with utilities to visualize (e.g. point clouds)'''
import open3d as o3d


def show_point_cloud(coords):
    """
    Shows point cloud in 3D

    Parameters
    ----------
    coords : Numpy array
        3D coordinates of cloud points

    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd.paint_uniform_color([1, 0, 0])])


# Print iterations progress
def progressbar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#', print_end=''):
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------
    iteration : int
        Required. Current iteration
    total : int
        Required. Total iterations
    prefix : string
        Prefix string
    suffix : string
        Suffix string
    decimals : int
        Positive number of decimals in percent complete
    length : int
        Character length of bar
    fill : string
        Bar fill character
    print_end : string
        (e.g. "\r", "\r\n")

    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
