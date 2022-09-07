''' Code with utilities to interpolate coordinates between two countours'''

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interpn


def ndgrid(*args, **kwargs):
    """
    Same as calling numpy ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)


def bwperim(bw, n=4):
    """
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.

    Parameters
    ----------
    bw : Numpy array
        A binary image
    n : int
        Connectivity. Must be 4 or 8 (default: 4)


    Returns
    -------
    perim : Numpy array
        A boolean image


    """

    if n not in (4, 8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows, cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows, cols))
    south = np.zeros((rows, cols))
    west = np.zeros((rows, cols))
    east = np.zeros((rows, cols))

    north[:-1, :] = bw[1:, :]
    south[1:, :] = bw[:-1, :]
    west[:, :-1] = bw[:, 1:]
    east[:, 1:] = bw[:, :-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west == bw) & \
          (east == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:] = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:] = bw[:-1, :-1]
        south_west[1:, :-1] = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw


def signedbwdist(im):
    """
    Find perim and return masked image (signed/reversed)

    Parameters
    ----------
    im : Numpy array
        A binary image


    Returns
    -------
    im : Numpy array
        A binary image

    """

    im = -bwdist(bwperim(im)) * np.logical_not(im) + bwdist(bwperim(im)) * im
    return im


def bwdist(im):
    """
    Find euclidean distance map of image

    Parameters
    ----------
    im : Numpy array
        A binary image


    Returns
    -------
    im : Numpy array
        A binary image

    """
    dist_im = distance_transform_edt(1 - im)
    return dist_im


def interpshape(top, bottom, position):
    """
    Find interpolated shape at specific position between top and bottom

    Parameters
    ----------
    top : Numpy array
        Top binary image

    bottom : Numpy array
        Bottom binary image

    position: float
        Position between images where the interpolation is required.
        E.g. position=0.5 - Interpolates the middle image between top and bottom image


    Returns
    -------
    out : Numpy array
        Interpolated binary image

    """

    if position > 1.0:
        print("Error: Position must be between 0 and 1 (float)")

    top = signedbwdist(top)
    bottom = signedbwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))
    # create ndgrids
    points = (np.r_[0, 1], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r * c, 2))
    xi = np.c_[np.full((r * c), position), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out




