import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tifffile


#% INTERACTIVE VISUALIZATION FUNCTIONS - DO NOT WORK WITH INLINE FIGURES
def arrow_navigation(event,z,Z):
    '''
    Change z using arrow keys for interactive inspection.
    @author: vand at dtu dot dk
    '''
    if event.key == "up":
        z = min(z+1,Z-1)
    elif event.key == 'down':
        z = max(z-1,0)
    elif event.key == 'right':
        z = min(z+10,Z-1)
    elif event.key == 'left':
        z = max(z-10,0)
    elif event.key == 'pagedown':
        z = min(z+50,Z+1)
    elif event.key == 'pageup':
        z = max(z-50,0)
    return z


def inspect_vol(V, cmap=plt.cm.gray, vmin = None, vmax = None):
    """
    Inspect volumetric data.
    
    Parameters
    ----------
    V : 3D numpy array, it will be sliced along axis=0.  
    cmap : matplotlib colormap
        The default is plt.cm.gray.
    vmin and vmax: float
        color limits, if None the values are estimated from data.
        
    Interaction
    ----------
    Use arrow keys to change a slice.
    
    @author: vand at dtu dot dk
    """
    def update_drawing():
        ax.images[0].set_array(V[z])
        ax.set_title(f'slice z={z}/{Z}')
        fig.canvas.draw()

    def key_press(event):
        nonlocal z
        z = arrow_navigation(event,z,Z)
        update_drawing()

    Z = V.shape[0]
    z = (Z-1)//2
    fig, ax = plt.subplots()
    if vmin is None:
        vmin = np.min(V)
    if vmax is None:
        vmax = np.max(V)
    ax.imshow(V[z], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'slice z={z}/{Z}')
    fig.canvas.mpl_connect('key_press_event', key_press)
    
    
def inspect_tifvol(filename, cmap=plt.cm.gray, vmin = None, vmax = None):
    ''' 
    Inspect volume saved as tif stack or collection of tifs.

    Parameters
    ----------
    filename : str
        A name of a stacked tif file or a name of a folder containing a
        collection of tif files.
    cmap : matplotlib colormap
        The default is plt.cm.gray.
    vmin and vmax: float
        color limits, if None the values are estimated from the middle slice.
        
    Interaction
    ----------
    Use arrow keys to change a slice.
 
    Author: vand@dtu.dk, 2021
    '''

    def update_drawing():
        I = readslice(z)
        ax.images[0].set_array(I)
        ax.set_title(f'slice z={z}/{Z}')
        fig.canvas.draw()

    def key_press(event):
        nonlocal z
        z = arrow_navigation(event,z,Z)
        update_drawing()

    if os.path.isdir(filename):
        D = sorted(glob.glob(filename + '/*.tif*'))
        Z = len(D)
        readslice = lambda z: tifffile.imread(D[z])
    else:
        tif = tifffile.TiffFile(filename)
        Z = len(tif.pages)
        readslice = lambda z: tifffile.imread(filename, key = z)
      
    z = (Z-1)//2
    I = readslice(z)
    fig, ax = plt.subplots()
    if vmin is None:
        vmin = np.min(I)
    if vmax is None:
        vmax = np.max(I)
    ax.imshow(I, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'slice z={z}/{Z}')
    fig.canvas.mpl_connect('key_press_event', key_press)
    
    
def load_tifvol(filename, sub=None):
    ''' 
    Load volume from tif stack or collection of tifs.

    Parameters
    ----------
    filename : str
        A name of a stacked tif file or a name of a folder containing a
        collection of tif files.
    sub : a list containing three array-likes with the slices to be loaded from
        each of the three dimendions.
    
    Returns
    -------
    3D numpy array.

    
    Author: vand@dtu.dk, 2021
    '''
  
    if os.path.isdir(filename):
        D = sorted(glob.glob(filename + '/*.tif*'))
        Z = len(D)
        readslice = lambda z: tifffile.imread(D[z])
    else:
        tif = tifffile.TiffFile(filename)
        Z = len(tif.pages)
        readslice = lambda z: tifffile.imread(filename, key = z)
      
    oneimage = readslice(0)
    dim = (Z,) + oneimage.shape 
    
    if sub is None:
        sub = [None, None, None]
    for i in range(3):
        if sub[i] is None:
            sub[i] = np.arange(dim[i])
        sub[i] = np.asarray(sub[i]) # ensure np as we reshape later
    
    V = np.empty((len(sub[0]), len(sub[1]), len(sub[2])), dtype=oneimage.dtype)
    
    for i in range(len(sub[0])):
        I = readslice(sub[0][i])
        V[i] = I[sub[1].reshape((-1,1)), sub[2].reshape((1,-1))]
    
    return V
    

def save_tifvol(V, filename, stacked=True):
    '''
    Saves tifvol using tifffile. 
    Does not (yet) support resolution and xy axis flip.

    Parameters
    ----------
    V : 3D numpy array, it will saved in slices along axis=0.
    filename : str with filename.
    stacked : bool, default is True.
        Whether to save one stacked tif file or a collection of tifs.


    '''
    if stacked:
        tifffile.imwrite(filename, V[0])
        for  z in range(1, V.shape[0]):
            tifffile.imwrite(filename, V[z], append=True)
    else:
        nr_digits = len(str(V.shape[0]))
        nr_format = '{:0' + str(nr_digits) + 'd}'
        for z in range(V.shape[0]):
            tifffile.imwrite(filename + nr_format.format(z) + '.tif', V[z])
 