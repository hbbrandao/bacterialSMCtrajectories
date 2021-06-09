import matplotlib.pyplot as plt
import numpy as np
import os

from multiprocessing import Pool
import pickle

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

import scipy

def contact_map_generator_from_SMC_list(smc_pairs,N=4000,base=10,nsamples=80):
    """
    Parameters
    ----------
    smc_pairs : list of tuples
        List of 2-tuples containing locations of SMCs on the chain of length `N`

    N : int
        Length of the polymer chain.

    base : int
        Division value to convert chain position to index position on `HiC` heatmap

    Returns
    -------

    HiC : ndarray
        Contact proabilities from sampled points of smc_pairs conformation

    """

    row = [x for x in range(N) ] + [x[0] for x in smc_pairs]
    col = [np.mod(x+1,N) for x in range(N)] + [x[1] for x in smc_pairs]
    deff = shortest_path(csr_matrix((np.ones(len(row)),(row,col)),shape=(N,N)),directed=False)
    deff = zoomArray(deff,(N//base,N//base))

    HiC = 1/np.sqrt(deff+1)**3

    return HiC

def contact_map_generator_from_SMC_list_slow(smc_pairs,N=4000,base=10,nsamples=80):
    """
    Parameters
    ----------
    smc_pairs : list of tuples
        List of 2-tuples containing locations of SMCs on the chain of length `N`

    N : int
        Length of the polymer chain.

    base : int
        Division value to convert chain position to index position on `HiC` heatmap

    Returns
    -------

    HiC : ndarray
        Contact proabilities from sampled points of smc_pairs conformation

    HiC_count : ndarray
        Number of sampled points per HiC map position

    """

    row = [x for x in range(N) ] + [x[0] for x in smc_pairs]
    col = [np.mod(x+1,N) for x in range(N)] + [x[1] for x in smc_pairs]
    deff = shortest_path(csr_matrix((np.ones(len(row)),(row,col)),shape=(N,N)),directed=False)


    # create heatmap
    HiC = np.zeros((N//base,N//base))
    HiC_count = np.zeros((N//base,N//base))

    vals = sorted(np.random.choice(N,nsamples, replace=False))
    for ix in range(len(vals)):
        for iy in range(ix,len(vals)):
            x = vals[ix]
            y = vals[iy]
            if deff[x,y] == 0:
                pc = 1
            else:
                pc = 1/np.sqrt(deff[x,y])**3

            if not np.isnan(pc):
                HiC_count[x//base,y//base] += 1
                HiC_count[y//base,x//base] += 1
                HiC[x//base,y//base] += pc
                HiC[y//base,x//base] += pc
    return HiC, HiC_count


class heatmap():
    def __init__(self,HiC,HiC_counts,nsamples=80):
        self.HiC = HiC
        self.HiC_count = HiC_counts
        self.numFailed = 0
        self.totsamp = 0
        self.nsamples = nsamples

    def add(self,hmap):
        self.HiC += hmap.HiC
        self.HiC_count += hmap.HiC_count
        self.numFailed += hmap.numFailed
        self.totsamp += hmap.totsamp

        
def zoomArray(inArray, finalShape, sameSum=False,
              zoomFunction=scipy.ndimage.zoom, **zoomKwargs):
    """
    This function originated from mirnylib.numutils.
    https://github.com/mirnylab/mirnylib-legacy/blob/hg/mirnylib/numutils.py
    
    It is included here for convenience.
    
    Description
    -----------

    Normally, one can use scipy.ndimage.zoom to do array/image rescaling.
    However, scipy.ndimage.zoom does not coarsegrain images well. It basically
    takes nearest neighbor, rather than averaging all the pixels, when
    coarsegraining arrays. This increases noise. Photoshop doesn't do that, and
    performs some smart interpolation-averaging instead.

    If you were to coarsegrain an array by an integer factor, e.g. 100x100 ->
    25x25, you just need to do block-averaging, that's easy, and it reduces
    noise. But what if you want to coarsegrain 100x100 -> 30x30?

    Then my friend you are in trouble. But this function will help you. This
    function will blow up your 100x100 array to a 120x120 array using
    scipy.ndimage zoom Then it will coarsegrain a 120x120 array by
    block-averaging in 4x4 chunks.

    It will do it independently for each dimension, so if you want a 100x100
    array to become a 60x120 array, it will blow up the first and the second
    dimension to 120, and then block-average only the first dimension.

    Parameters
    ----------

    inArray: n-dimensional numpy array (1D also works)
    finalShape: resulting shape of an array
    sameSum: bool, preserve a sum of the array, rather than values.
             by default, values are preserved
    zoomFunction: by default, scipy.ndimage.zoom. You can plug your own.
    zoomKwargs:  a dict of options to pass to zoomFunction.
    """
    inArray = np.asarray(inArray, dtype=np.double)
    inShape = inArray.shape
    assert len(inShape) == len(finalShape)
    mults = []  # multipliers for the final coarsegraining
    for i in range(len(inShape)):
        if finalShape[i] < inShape[i]:
            mults.append(int(np.ceil(inShape[i] / finalShape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    tempShape = tuple([i * j for i, j in zip(finalShape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoomMultipliers = np.array(tempShape) / np.array(inShape) + 0.0000001
    assert zoomMultipliers.min() >= 1

    # applying scipy.ndimage.zoom
    rescaled = zoomFunction(inArray, zoomMultipliers, **zoomKwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1:]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)
    assert rescaled.shape == finalShape

    if sameSum:
        extraSize = np.prod(finalShape) / np.prod(inShape)
        rescaled /= extraSize
    return rescaled        