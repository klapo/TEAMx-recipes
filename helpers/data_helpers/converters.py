import numpy as np


def distance_convert(ds, swap_dim):
    """ Create a "distance" coordinate to sample over instead of "x" and "y".

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with the dimensions "x" and "y"
    swap_dim : str
        Indicates the dimension that labels the derived distance coordinate. The
        function swaps from this coordinate to the distance coordinate.

    Returns
    -------
    ds : xarray.Dataset
        The original dataset with the "distance" coordinate now labeling the data.

    """
    x1 = ds.x.values[0]
    y1 = ds.y.values[0]
    x2 = ds.x.values[-1]
    y2 = ds.y.values[-1]

    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = np.linspace(0, d, ds.LAF.size)
    ds.coords['distance'] = (swap_dim, distance)
    ds = ds.swap_dims({swap_dim: 'distance'})

    return ds
