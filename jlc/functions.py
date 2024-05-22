import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import PIL
import torch
import cv2
import copy 
import random
from PIL import Image, ImageDraw, ImageFont
from pprint import pprint
import matplotlib
import io
from matplotlib.patheffects import withStroke
from tempfile import NamedTemporaryFile
import os
from . import nc
import warnings
from skimage.measure import find_contours

def montage(arr,
            maintain_aspect=True,
            reshape=True,
            text=None,
            return_im=False,
            imshow=True,
            reshape_size=None,
            n_col=None,
            n_row=None,
            padding=0,
            padding_color=0,
            rows_first=True,
            figsize_per_pixel=1/100,
            text_color=[0,0,0],
            text_size=10,
            create_figure=True):
    """
    Displays and returns an montage of images from a list or 
    list of lists of images.

    Parameters
    ----------
    arr : list
        A list or list of lists containing images (np.arrays) of shape 
        (d1,d2), (d1,d2,1), (d1,d2,3) or (d1,d2,4). If arr is a list of lists
        then the first list dimensions is vertical and second is horizontal.
        If there is only one list dimensions then the list will be put in an
        appropriate 2D grid of images. The input can also be a 5D or 4D 
        np.array and in this case the first two dimensions are intepreted 
        the same way as if they were a list. Even if the 5th channel dimension
        is size 1 it has to in included in this case.
    maintain_aspect : boolean, optional
        Should image aspect ratios be maintained. Only relevant if 
        reshape=True. The default is True.
    reshape : boolean, optional
        Should images be reshaped to better fit the montage image. The default 
        is True.
    imshow : boolean, optional
        Should plt.imshow() be used inside the function. The default is True.
    reshape_size : array-like, optional
        2 element list or array like variable. Specifies the number of pixels 
        in the first dim (vertical) and second dim (horizontal) per image in
        the resulting concatenated image
        The default is None.
    n_col : int, optional
        Number of columns the montage will contain.
        The default is None.
    n_row : int, optional
        Number of rows the montage will contain.
        The default is None.
    padding : int or [int,int], optional
        Number of added rows/columns of padding to each image. If an int is
        given the same horizontal and vertical padding is used. If a list is
        given then the first index is the number of vertical padding pixels and
        the second index is the number of horizontal padding pixels. 
        The default is None.
    padding_color : float or int
        The color of the used padding. The default is black (0).
    rows_first : bool
        If True and a single list is given as arr then the images will first
        be filled into row 0 and then row 1, etc. Otherwise columns will be
        filled first. The default is True.
    figsize_per_pixel : float
        How large a figure to render if imshow=True, in relation to pixels.
        Defaults to 1/100.
    text_color : matplotlib color-like
        color of text to write on top of images. Defaults to red ([1,0,0]).
    text_size : float or int
        Size of text to write on top of images. Defaults to 10.
    create_figure : bool
        Should plt.figure() be called when imshow is True? Defaults to True.
    Returns
    -------
    im_cat : np.array
        Concatenated montage image.
        
    
    Example
    -------
    montage(np.random.rand(2,3,4,5,3),reshape_size=(40,50))

    """
    if torch.is_tensor(arr):
        assert len(arr.shape)==4, "torch tensor must have at 4 dims, formatted as (n_images,channels,H,W)"
        arr = arr.detach().cpu().clone().permute(0,2,3,1).numpy()
    if isinstance(arr,np.ndarray):
        if len(arr.shape)==4:
            arr = [arr[i] for i in range(arr.shape[0])]
        elif len(arr.shape)==5:
            n1 = arr.shape[0]
            n2 = arr.shape[1]
            arr = [[arr[i,j] for j in range(arr.shape[1])]
                   for i in range(arr.shape[0])]
        else:
            raise ValueError("Cannot input np.ndarray with less than 4 dims")
    
    if isinstance(arr[0],np.ndarray): #if arr is a list or 4d np.ndarray
        if (n_col is None) and (n_row is None):
            n1 = np.floor(len(arr)**0.5).astype(int)
            n2 = np.ceil(len(arr)/n1).astype(int)
        elif (n_col is None) and (n_row is not None):
            n1 = n_row
            n2 = np.ceil(len(arr)/n1).astype(int)
        elif (n_col is not None) and (n_row is None):
            n2 = n_col
            n1 = np.ceil(len(arr)/n2).astype(int)
        elif (n_col is not None) and (n_row is not None):
            assert n_col*n_row>=len(arr), "number of columns/rows too small for number of images"
            n1 = n_row
            n2 = n_col
        
        if rows_first:
            arr2 = []
            for i in range(n1):
                arr2.append([])
                for j in range(n2):
                    ii = n2*i+j
                    if ii<len(arr):
                        arr2[i].append(arr[ii])
        else:
            arr2 = [[] for _ in range(n1)]
            for j in range(n2):
                for i in range(n1):
                    ii = i+j*n1
                    if ii<len(arr):
                        arr2[i].append(arr[ii])
        arr = arr2
    if n_row is None:
        n1 = len(arr)
    else:
        n1 = n_row
        
    n2_list = [len(arr[i]) for i in range(n1)]
    if n_col is None:
        n2 = max(n2_list)
    else:
        n2 = n_col
        
    idx = []
    for i in range(n1):
        idx.extend([[i,j] for j in range(n2_list[i])])
    n = len(idx)
    idx = np.array(idx)
    
    N = list(range(n))
    I = idx[:,0].tolist()
    J = idx[:,1].tolist()
    
    D1 = np.zeros(n,dtype=int)
    D2 = np.zeros(n,dtype=int)
    aspect = np.zeros(n)
    im = np.zeros((32,32,3))
    channels = 1
    for n,i,j in zip(N,I,J): 
        if arr[i][j] is None:#image is replaced with zeros of the same size as the previous image
            arr[i][j] = np.zeros_like(im)
        else:
            assert isinstance(arr[i][j],np.ndarray), "images in arr must be np.ndarrays (or None for a zero-image)"
        im = arr[i][j]
        
        D1[n] = im.shape[0]
        D2[n] = im.shape[1]
        if len(im.shape)>2:
            channels = max(channels,im.shape[2])
            assert im.shape[2] in [1,3,4]
            assert len(im.shape)<=3
    aspect = D1/D2
    if reshape_size is not None:
        G1 = reshape_size[0]
        G2 = reshape_size[1]
    else:
        if reshape:
            G2 = int(np.ceil(D2.mean()))
            G1 = int(np.round(G2*aspect.mean()))
        else:
            G1 = int(D1.max())
            G2 = int(D2.max())
    if padding is not None:
        if isinstance(padding,int):
            padding = [padding,padding]
    else:
        padding = [0,0]
        
    p1 = padding[0]
    p2 = padding[1]
    G11 = G1+p1*2
    G22 = G2+p2*2
    
    
    im_cat_size = [G11*n1,G22*n2]

    im_cat_size.append(channels)
    im_cat = np.zeros(im_cat_size)
    if channels==4:
        im_cat[:,:,3] = 1

    for n,i,j in zip(N,I,J): 
        im = arr[i][j]
        if issubclass(im.dtype.type, np.integer):
            im = im.astype(float)/255
        if not reshape:
            d1 = D1[n]
            d2 = D2[n]
        else:
            z_d1 = G1/D1[n]
            z_d2 = G2/D2[n]
            if maintain_aspect:
                z = [min(z_d1,z_d2),min(z_d1,z_d2),1][:len(im.shape)]
            else:
                z = [z_d1,z_d2,1][:len(im.shape)]
            im = nd.zoom(im,z)
            d1 = im.shape[0]
            d2 = im.shape[1]
            
        if len(im.shape)==3:
            im = np.pad(im,((p1,p1),(p2,p2),(0,0)),constant_values=padding_color)
        elif len(im.shape)==2:
            im = np.pad(im,((p1,p1),(p2,p2)),constant_values=padding_color)
        else:
            raise ValueError("images in arr must have 2 or 3 dims")
            
        d = (G1-d1)/2
        idx_d1 = slice(int(np.floor(d))+i*G11,G11-int(np.ceil(d))+i*G11)
        d = (G2-d2)/2
        idx_d2 = slice(int(np.floor(d))+j*G22,G22-int(np.ceil(d))+j*G22)
        
        if len(im.shape)>2:
            im_c = im.shape[2]
        else:
            im_c = 1
            im = im[:,:,None]
            
        if im_c<channels:
            if channels>=3 and im_c==1:
                if len(im.shape)>2:
                    im = im[:,:,0]
                im = np.stack([im]*3,axis=2)
            if channels==4 and im_c<4:
                im = np.concatenate([im]+[np.ones((im.shape[0],im.shape[1],1))],axis=2)
        im_cat[idx_d1,idx_d2,:] = im
    #im_cat = np.clip(im_cat,0,1)
    if imshow:
        if create_figure:
            plt.figure(figsize=(figsize_per_pixel*im_cat.shape[1],figsize_per_pixel*im_cat.shape[0]))
        
        is_rgb = channels>=3
        if is_rgb:
            plt.imshow(np.clip(im_cat,0,1),vmin=0,vmax=1)
        else:
            plt.imshow(im_cat,cmap="gray")

        if text is not None:
            #max_text_len = max([max(list(map(len,str(t).split("\n")))) for t in text])
            #text_size = 10#*G22/max_text_len*figsize_per_pixel #42.85714=6*16/224/0.01
            for i,j,t in zip(I,J,text):
                dt1 = p1+G11*i
                dt2 = p2+G22*j
                plt.text(x=dt2,y=dt1,s=str(t),color=text_color,va="top",ha="left",size=text_size)

    if return_im:
        return im_cat

def cat(arrays,axis=0,new_dim=False):
    """
    Very unsafe concatenation of arrays.
    ------------------------------------------
    Inputs:
        arrays : list or tuple of np.ndarrays
            Arrays to concatenate. If the arrays do not have the same size over
            any of the non concatenation axis dimensions, then the arrays will
            be tiled to fit the maximum size over the dimension. e.g. if we 
            concatenate a np.size([1,5,2]) and np.size([3,5,4]) over axis=0
            then the output will have size np.size([3+1,5,4]) and the first
            array will be repeated twice over dimension 2.
        axis : int, optional
            Axis over which the main concatenation should be done over. The 
            default is 0.
        new_dim : bool, optional
            Should a new axis be inserted over the concatenation dim? The 
            default is False.
    Outputs:
        cat_arrays : np.ndarray
            The concatenated array
    """
    n_dims = np.array([len(np.array(array).shape) for array in arrays]).max()
    if n_dims<axis:
        n_dims = axis
    cat_arrays = []
    for array in arrays:
        if np.size(array)>1:
            tmp = np.array(array).copy()
            tmp = np.expand_dims(tmp,axis=tuple(range(len(tmp.shape),n_dims)))
            cat_arrays.append(tmp)
    if new_dim or len(cat_arrays[0].shape)<=axis:
        for i in range(len(cat_arrays)):
            cat_arrays[i] = np.expand_dims(cat_arrays[i],axis=axis)
    SHAPE = np.array([list(array.shape) for array in cat_arrays]).max(0)
    for i in range(len(cat_arrays)):
        reps = SHAPE//(cat_arrays[i].shape)
        reps[axis] = 1
        cat_arrays[i] = np.tile(cat_arrays[i],reps)
    cat_arrays = np.concatenate(cat_arrays,axis=axis)
    return cat_arrays

def reverse_dict(dictionary,check_for_duplicates=False):
    """
    Function to reverse a dictionary.
    ---------------------------------
    Inputs:
        dictionary : dict
            Dictionary to be inverted such that keys and values are swapped.
        check_for_duplicates : bool, optional
            If True then the function throws an error if there are duplicates 
            in the values of the dict. The default is False.
    Outputs:
        inv_dict : dict
            The inverted dictionary.
    """
    inv_dict = {}
    values = []
    for k,v in dictionary.items():
        if check_for_duplicates:
            assert v not in values,"Found duplicate value in dict, cannot invert"
            values.append(v)
        inv_dict[v] = k
    return inv_dict
    
def num_of_params(model,print_numbers=True):
    """
    Prints and returns the number of paramters for a pytorch model.
    Args:
        model (torch.nn.module): Pytorch model which you want to extract the number of 
        trainable parameters from.
        print_numbers (bool, optional): Prints the number of parameters. Defaults to True.

    Returns:
        n_trainable (int): Number of trainable parameters.
        n_not_trainable (int): Number of not trainable parameters.
        n_total (int): Total parameters.
    """
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_not_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_total = n_trainable+n_not_trainable
    if print_numbers:
        s = ("The model has:"
            +"\n"+str(n_trainable)+" trainable parameters"
            +"\n"+str(n_not_trainable)+" untrainable parameters"
            +"\n"+str(n_total)+" total parameters")
        print(s)
    return n_trainable,n_not_trainable,n_total


def standardize_image_like(im,n_output_channels=3,return_type="np",dtype="float64"):
    """
    Takes torch,np or pil image objects and makes it have 
    standardized dtype, order of channels, length of dims and return type

    Args:
        im (_type_): _description_
        n_output_channels (int, optional): _description_. Defaults to 3.
        return_type (str, optional): _description_. Defaults to "np".

    Returns:
        _type_: _description_
    """
    im = copy.copy(im)
    assert return_type.lower() in ["np","torch","pil"]
    assert n_output_channels in [0,1,3,4]
    pil_types = [PIL.Image.Image,
                 PIL.PngImagePlugin.PngImageFile]
    
    if type(im) in pil_types:
        im = np.array(im)
    elif torch.is_tensor(im):
        im = im.detach().cpu().numpy()
    else:
        assert isinstance(im,np.ndarray)
    
    if len(im.shape)>3:
        im = im.squeeze()
        assert len(im.shape)<=3, "images must have at most 3 non-singleton dimensions"

    im = np.atleast_3d(im)

    if im.shape[2] in [1,3,4]:
        1
    elif im.shape[0] in [1,3,4]:
        im = np.transpose(im,axes=[1,2,0]) 
    else:
        assert im.shape[1] in [1,3,4], "The image has to have 1,3 or 4 color channels in one of the 3 dimensions. im.shape="+str(im.shape)
        im = np.transpose(im,axes=[0,2,1])

    uint8_out = (dtype=="uint8" or dtype==np.dtype("uint8"))

    if im.dtype==np.dtype('uint8') and im.max()>1 and not uint8_out:
        im = im.astype(dtype)/255
    else:
        im = im.astype(dtype)


    if im.min()<0:
        im = im-(im.min())
    if im.max()>=1:
        im = im/(im.max())

    if n_output_channels==0:
        im = im.mean(2)
    elif n_output_channels==1:
        im = im.mean(2,keepdims=True)
    elif n_output_channels==3:
        if im.shape[2]==1:
            im = np.tile(im,(1,1,3))
        elif im.shape[2]==4:
            im = im[:,:,:3]
        else:
            assert im.shape[2]==3
    elif n_output_channels==4:
        if im.shape[2]==1:
            im = np.tile(im,(1,1,3))
            im = np.concatenate((im,np.ones((im.shape[0],im.shape[1],1))),axis=2)
        elif im.shape[2]==3:
            im = np.concatenate((im,np.ones((im.shape[0],im.shape[1],1))),axis=2)
        else:
            assert im.shape[2]==4
    
    if return_type=="np":
        pass
    elif return_type=="torch":
        im = torch.from_numpy(im)
    elif return_type=="pil":
        im = PIL.Image.fromarray(im.astype("uint8"))
    return im


def collage(arr,
            imshow=True,
            return_image=False,
            align_rows=True,
            n_alignment=None,
            alignment_size=None,
            random_order=True,
            exact_fit_downsize=True,
            border_color=[0,0,0],
            figsize_per_pixel=1/100):
    """A function to make a nice square image out of a list of images.

    Args:
        arr (list): list of images with len(arr)=n (torch,numpy or PIL) that 
            should be turned into a collage. Can also be a 4D torch tensor or
            numpy array were the 0th axis is has dim n.
        imshow (bool, optional): should the image be shown (with plt.imshow). 
            Defaults to True.
        return_image (bool, optional): Should the function return 
            the collage image. Defaults to False.
        align_rows (bool, optional): images are aligned in either the row 
            direction (if True) or the column direction (if False). Defaults 
            to True.
        n_alignment (int, optional): How many rows or columns should the 
            images be alignet into. Defaults to n_align=int(np.floor(np.sqrt(len(arr)))).
        alignment_size (int, optional): How many pixels should constitute each 
            aligned row of images have in the opposite direction than the alignment 
            direction. Defaults to either the minimum of the images, or the median
            if the minimum is significantly different from the minimum.
        random_order (bool, optional): Should images be placed randomly in the
            collage (if True) or in a alignment-by-alignment fashion (if False). 
            Defaults to True.
        exact_fit_downsize (bool, optional): Should each alignment row/column of 
            images be resized such that there is no bordering pixels without images. 
            Defaults to True.
        border_color (list, optional): What RGB value should pixels assume where 
            there is no images. Only relevant if exact_fit_downsize=False. Defaults 
            to [0,0,0].
        figsize_per_pixel (_type_, optional): How large should the figure be if 
            imshow=True, measured in matplotlib.pyplot figsizes per pixels in the 
            final collage image. Defaults to 1/100.

    Returns:
        collage_im np.array with size (x,y,3): The original n images from the arr
            variable as the collage image. Only returned if return_image=True.
    """
    if isinstance(arr,np.ndarray) or torch.is_tensor(arr):
        if len(arr.shape)<=4:
            arr = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError("Cannot input np.ndarray with more than 4 dims")
    
    assert isinstance(arr,list), "arr must be list, np.ndarray or torch.tensor"
    for i in range(len(arr)):
        arr[i] = standardize_image_like(arr[i])

    align_dim = 0 if align_rows else 1
    off_dim = 1 if align_rows else 0
    n = len(arr)

    if n_alignment is None:
        n_alignment = int(np.floor(np.sqrt(n)))

    if random_order:
        order = np.random.permutation(n)
        arr = [arr[i] for i in order]
    
    shapes = np.array([a.shape for a in arr])[:,:2]
    
    shapes_resized = shapes.copy()
        
    shapes_resized = shapes_resized/(shapes_resized[:,align_dim,None])
    if alignment_size is None:    
        alignment_size = shapes[:,align_dim].min()
        if np.quantile(shapes[:,align_dim],0.5)*0.5>alignment_size:
            alignment_size = int(np.ceil(np.quantile(shapes[:,align_dim],0.5)))

    shapes_resized = np.round(alignment_size*shapes_resized).astype(int)

    
    if random_order:
        pixels_per_align = [0 for _ in range(n_alignment)]
        idx_per_align = [[] for _ in range(n_alignment)]
        for idx in range(n):
            bin_sample = np.flatnonzero([min(pixels_per_align)==p for p in pixels_per_align])
            i = np.random.choice(bin_sample)
            idx_per_align[i].append(idx)
            pixels_per_align[i] += shapes_resized[idx,off_dim]
    else:
        if n%n_alignment==0:
            num_per_align = n//n_alignment
            idx_per_align = [[j+i*num_per_align for j in range(num_per_align)] for i in range(n_alignment)]
            pixels_per_align = [sum([shapes_resized[idx,off_dim] for idx in idx_list]) for idx_list in idx_per_align]
        else:
            val_below = np.floor(n/n_alignment).astype(int)
            val_above = val_below+1
            num_above = (n%n_alignment)
            num_below = n_alignment-num_above
            num_per_bin = [val_below for _ in range(num_below)]+[val_above for _ in range(num_above)]
            best_cost = float("inf")
            n_tries = min(10,2**n_alignment)
            for _ in range(n_tries):
                random.shuffle(num_per_bin)
                idx_per_align_tmp = []
                k = 0
                for npb in num_per_bin:
                    idx_per_align_tmp.append(list(range(k,k+npb)))
                    k += npb
                pixels_per_align_tmp = [sum([shapes_resized[idx,off_dim] for idx in idx_list]) for idx_list in idx_per_align_tmp]
                cost = max(pixels_per_align_tmp)-min(pixels_per_align_tmp)
                if cost<best_cost:
                    best_cost = copy.copy(cost)
                    pixels_per_align = copy.copy(pixels_per_align_tmp)
                    idx_per_align = copy.copy(idx_per_align_tmp)
    alignment_im0 = np.zeros((alignment_size,0,3) if align_rows else (0,alignment_size,3))

    alignment_images = []
    for i in range(n_alignment):
        alignment_im = alignment_im0.copy()
        for image_i in idx_per_align[i]:
            im = arr[image_i]
            resize_shape = (shapes_resized[image_i,1],shapes_resized[image_i,0])
            im = cv2.resize(im, resize_shape, interpolation=cv2.INTER_AREA)
            alignment_im = np.concatenate((alignment_im,im),axis=off_dim)
        alignment_images.append(alignment_im)

    if exact_fit_downsize:
        collage_im = np.zeros((0,min(pixels_per_align),3) if align_rows else (min(pixels_per_align),0,3))
        for i in range(n_alignment):
            reshape_size = [int(np.round(alignment_size*min(pixels_per_align)/pixels_per_align[i])),
                            min(pixels_per_align)]
            if align_rows:
                reshape_size = [reshape_size[1],reshape_size[0]]

            im = cv2.resize(alignment_images[i], reshape_size, interpolation=cv2.INTER_AREA)
            collage_im = np.concatenate((collage_im,im),axis=align_dim)
    else:
        collage_im = np.zeros((0,max(pixels_per_align),3) if align_rows else (max(pixels_per_align),0,3))
        border_color = np.array(border_color).flatten().reshape(1,1,3)
        for i in range(n_alignment):
            if align_rows:
                im_border_size = [alignment_size,max(pixels_per_align)-pixels_per_align[i],3]
            else:
                im_border_size = [alignment_size,max(pixels_per_align)-pixels_per_align[i],3]
            im_border = border_color*np.ones(im_border_size)
            im = np.concatenate((alignment_images[i],im_border),axis=off_dim)
            collage_im = np.concatenate((collage_im,im),axis=align_dim)
    collage_im[collage_im>1] = 1
    if imshow:
        plt.figure(figsize=(figsize_per_pixel*collage_im.shape[1],figsize_per_pixel*collage_im.shape[0]))
        plt.imshow(collage_im,cmap="gray")
        plt.show()
    if return_image:
        return collage_im
    
def earth_mover_distance(vec1, vec2, input_is_raw_data=False):
    """Computes the earth mover distance (EM) between two vectors 
    each representing a discrete distribution. 

    Args:
        vec1 (list): The first vector. E.g lets say the distribution is 
            number of kids in families and the first vector represents 
            2 families, each with one kid. Then the vector should be
            vec1 = [2,0,0] (limit goes up to 3 kids in this case).
        vec2 (list): The second vector. E.g lets say the distribution is 
            number of kids in families and the second vector represents 
            2 families, with 2 and 3 kids. Then the vector should be
            vec1 = [0,1,1].
        input_is_raw_data (bool,Optional): If the input is raw data from 
            the discrete distribution. e.g. the data [0,1,1,2] would be 
            reconstructed into [1,2,1]. If the input is raw dete then
            the distribution need not be discrete. Defaults to False.
    Returns:
        dist: The measured earth mover distance.
    Example:
        vec1 = [2,0,0]
        vec2 = [0,1,1]
        distance = earth_mover_distance(vec1, vec2) 
        print(distance) #3
    """
    assert len(vec1) == len(vec2), "Input vectors must have the same length"
    if input_is_raw_data:
        idx1 = sorted(vec1)
        idx2 = sorted(vec2)
    else:
        assert sum(vec1) == sum(vec2), "Input vectors must have the same sum"
        idx1 = []
        idx2 = []
        for i in range(len(vec1)):
            idx1.extend([i for _ in range(vec1[i])])
            idx2.extend([i for _ in range(vec2[i])])
    dist = 0
    for i1,i2 in zip(idx1,idx2):
        dist += abs(i1-i2)
    dist = dist/len(idx1)
    return dist

def montage_save(save_name="test.png",
                 save_fig=True,
                 show_fig=True,
                 pixel_mult = 4,
                 **montage_kwargs
                ):
    """ Save a montage of images to a file (optional) and show it (optional)

    Args:
        save_name (str, optional): name of the file to save the image to. Defaults 
            to "test.png".
        save_fig (bool, optional): should the image be saved. Defaults to True.
        show_fig (bool, optional): should the image be shown. Defaults to True.
        pixel_mult (int, optional): Optional multiplier to be used to make pixels 
            consist of pixel_mult^2 pixels. Text shown on the montage can however
            use the lower resolution, making the text independent from how large
            or small the image is (best to use powers of 2 for nearest neighbour
            interpolation). Defaults to 4.
    """

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    montage_kwargs["create_figure"] = False
    montage_kwargs["return_im"] = True
    montage_im = montage(**montage_kwargs)
    fig.set_size_inches(montage_im.shape[1]*pixel_mult/100,montage_im.shape[0]*pixel_mult/100)
    if save_fig:
        fig.savefig(save_name)
    if show_fig:
        plt.show()
    plt.close(fig)


class DataloaderIterator():
    """
    Class which takes a pytorch dataloader and enables next() ad infinum and 
    self.partial_epoch gives an iterator which only iterates on a ratio of 
    an epoch 
    """
    def __init__(self,dataloader):
        """ initialize the dataloader wrapper
        Args:
            dataloader (torch.utils.data.dataloader.DataLoader): dataloader to sample from
        """
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)
        self.partial_flag = False
        self.partial_round_err = 0

    def __len__(self):
        return len(self.dataloader)
    
    def reset(self):
        """reset the dataloader iterator"""
        self.iter = iter(self.dataloader)
        self.partial_flag = False
        self.partial_round_err = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.partial_flag:
            if self.partial_counter==self.partial_counter_end:
                self.partial_flag = False
                raise StopIteration
        try:
            batch = next(self.iter)
        except StopIteration:    
            self.iter = iter(self.dataloader)
            batch = next(self.iter)
        if self.partial_flag:
            self.partial_counter += 1
        return batch

    def partial_epoch(self,ratio):
        """returns an iterable stopping after a partial epoch 
        Args:
            ratio (float): positive float denoting the ratio of the epoch.
                           e.g. 0.2 will give 20% of an epoch, 1.5 will
                           give one and a half epoch
        Returns:
            iterable: partial epoch iterable
        """
        self.partial_flag = True
        self.partial_counter_end_unrounded = len(self.dataloader)*ratio+self.partial_round_err
        self.partial_counter_end = int(round(self.partial_counter_end_unrounded))
        self.partial_round_err = self.partial_counter_end_unrounded-self.partial_counter_end
        self.partial_counter = 0
        if self.partial_counter_end==0:
            self.partial_counter_end = 1
        return iter(self)
    
class zoom:
    def __init__(self, ax=None):
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.cid_scroll = ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.zoom_factor = 1.2
        self.dragging = False
        self.prev_x = None
        self.prev_y = None

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        xdata, ydata = event.xdata, event.ydata

        if event.button == 'up':
            self.zoom_in(xdata, ydata)
        elif event.button == 'down':
            self.zoom_out(xdata, ydata)

        self.ax.figure.canvas.draw()
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            self.dragging = True
            self.prev_x = event.xdata
            self.prev_y = event.ydata

    def on_release(self, event):
        if event.button == 1:
            self.dragging = False
            self.prev_x = None
            self.prev_y = None

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return

        if self.dragging:
            if self.prev_x is not None and self.prev_y is not None:
                dx = event.xdata - self.prev_x
                dy = event.ydata - self.prev_y
                self.translate(dx, dy)
                self.ax.figure.canvas.draw()
            self.prev_x, self.prev_y = self.ax.transData.inverted().transform((event.x, event.y))

    def zoom_in(self, x, y):
        self.update(x, y, 1 / self.zoom_factor)

    def zoom_out(self, x, y):
        self.update(x, y, self.zoom_factor)

    def translate(self, dx, dy):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        xlim = xlim - dx
        ylim = ylim - dy

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.update()

    def update(self, x=None, y=None, factor=1.0):
        if x is not None and y is not None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            new_width = (xlim[1] - xlim[0]) * factor
            new_height = (ylim[1] - ylim[0]) * factor

            x_ratio = (x - xlim[0]) / (xlim[1] - xlim[0])
            y_ratio = (y - ylim[0]) / (ylim[1] - ylim[0])

            xlim = x - new_width * x_ratio, x + new_width * (1 - x_ratio)
            ylim = y - new_height * y_ratio, y + new_height * (1 - y_ratio)

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.ax.figure.canvas.draw()

valid_fontsize = lambda fontsize: max(1,np.round(fontsize).astype(int))

def get_bbox_params(text="A", fontname="times", 
                    return_im=False,
                    empty_ratio=0.1, 
                    height=256,
                    init_factor=0.5,
                    max_ite=10, 
                    ite_scale=2/3,
                    expected_w_h_ratio_per_letter=0.5,
                    post_buffer=0.1,
                    error_on_fail=True):
    """
    Function which returns a fontsize and position that 
    makes the text fit the frame exactly, in a normalized
    setting with y \in [0,1] and x \in [0,aspect_ratio]

    Inputs:
    - text: The text to be rendered
    - font: The font to be used
    ## hyperparameters
    - return_im: Whether to return the image with the text
    - empty_ratio: The ratio of the frame that should be 
                   empty before believing no text exists outside the frame
    - height: The height of the image used to render the text
    - init_factor: The initial factor to multiply the height by to get the fontsize
    - max_ite: The maximum number of iterations to find the right size
    - ite_scale: The scale of the step size for the iterations
    - expected_w_h_ratio_per_letter: The expected width to height ratio per letter
    - post_buffer: The buffer to leave after the text in the frame
    - error_on_fail: Whether to raise an error if the text does not fit the 
                    frame within max_ite iterations
    

    Returns (as a tuple):
    - fontsize: The fontsize that makes the text fit the frame
    - position: The position of the text in the frame
    - aspect_ratio: The aspect ratio of the frame
    """
    num_letters_width = max([len(line) for line in text.split("\n")])
    num_lines = text.count("\n")+1
    #size format is (width,height)=(x,y)=(dim 1,dim 0)
    size = (np.ceil(height*expected_w_h_ratio_per_letter*num_letters_width/num_lines).astype(int),height)
    image0 = Image.new("L", size, 0)
    fontsize = size[1]*init_factor
    position = (size[0] / 2, size[1] / 2)
    last_fontsize = copy.copy(fontsize)
    for ite in range(max_ite):
        image = image0.copy()
        draw = ImageDraw.Draw(image)
        draw.text(position, text, 255, font=ImageFont.truetype(fontname, valid_fontsize(fontsize)), anchor="mm")
        bbox = image.getbbox()
        current_empty_ratio = min(bbox[0]/size[0],
                                bbox[1]/size[1],
                                (size[0]-bbox[2])/size[0],
                                (size[1]-bbox[3])/size[1])
        accept = current_empty_ratio>=empty_ratio
        if accept:
            if last_fontsize<=fontsize:
                accept = False
                fontsize /= ite_scale
            else:
                break
        else:
            last_fontsize = copy.copy(fontsize)
            fontsize = fontsize*ite_scale
            
    if not accept:
        if error_on_fail:
            raise ValueError("Could not find a fontsize that makes the text fit the frame")
    fontsize = valid_fontsize(fontsize)
    #render a new image with the right fontsize, leaving exactly buffer of the height and width as margin on each side
    aspect_ratio = (bbox[2]-bbox[0]+1)/(bbox[3]-bbox[1]+1)
    post_size = (np.round(height*aspect_ratio).astype(int),height)
    aspect_ratio = post_size[1]/post_size[0]
    post_image = Image.new("L", post_size, 0)
    post_fontsize = valid_fontsize(post_size[1]/(bbox[3]-bbox[1]+1)*(1-post_buffer)*fontsize)
    new_bbox = get_new_bbox(bbox,scale=post_fontsize/fontsize,anchor=position)
    #draw old and new bbox on the image
    #draw.rectangle(bbox, outline=255)
    #draw.rectangle(new_bbox, outline=255)
    post_top_left = (post_size[0]*post_buffer/2,
                     post_size[1]*post_buffer/2)
    post_position = (position[0]+post_top_left[0]-new_bbox[0],
                     position[1]+post_top_left[1]-new_bbox[1])
    draw = ImageDraw.Draw(post_image)
    font = ImageFont.truetype(fontname, post_fontsize)
    draw.text(post_position, text, 255, font=font, anchor="mm")

    #repeat for final estimates with no buffer
    bbox = post_image.getbbox()
    aspect_ratio = (bbox[2]-bbox[0]+1)/(bbox[3]-bbox[1]+1)
    fontsize = valid_fontsize(post_fontsize*height/(bbox[3]-bbox[1]+1))
    new_bbox = get_new_bbox(bbox,scale=fontsize/post_fontsize,anchor=post_position)
    position = (post_position[0]-new_bbox[0],
                post_position[1]-new_bbox[1])
    
    if return_im:
        size = (np.round(height*aspect_ratio).astype(int),height)
        image = Image.new("L", size, 0)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(fontname, fontsize)
        draw.text(position, text, 255, font=font, anchor="mm")
        fontsize /= height
        position = (position[0]/height,position[1]/height)
        return fontsize,position,aspect_ratio,image
    else:
        fontsize /= height
        position = (position[0]/height,position[1]/height)
        return fontsize,position,aspect_ratio

def get_new_bbox(bbox,scale,anchor):
    """
    Function which returns the bounding box obtained by
    scaling the input bbox fixed on the anchor by the scale
    """
    return [anchor[0]+scale*(bbox[0]-anchor[0]),
            anchor[1]+scale*(bbox[1]-anchor[1]),
            anchor[0]+scale*(bbox[2]-anchor[0]),
            anchor[1]+scale*(bbox[3]-anchor[1])]

def render_text(text="A", fontname="times", size=(64, 64), factor=1, crop_to_bbox=False):
    if isinstance(size, int):
        size = (size, size)
    if size[0] is None:
        height = size[1]
    elif size[1] is None:
        height = size[0]
    else:
        height = size[1]
    fontsize,position,aspect_ratio = get_bbox_params(text, fontname, height=height, return_im=False)
    print(fontsize,position,aspect_ratio)
    is_too_wide = False
    is_too_tall = False
    if size[0] is None:
        size[0] = size[1]/aspect_ratio
    elif size[1] is None:
        size[1] = size[0]*aspect_ratio
    else:
        if size[0]/size[1]>aspect_ratio:
            if crop_to_bbox:
                size = (size[1]*aspect_ratio, size[1])
            else:
                is_too_wide = True
        elif size[0]/size[1]<aspect_ratio:            
            if crop_to_bbox:
                size = (size[0], size[0]/aspect_ratio)
            else:
                is_too_tall = True
        else:
            pass
    size_factor = (np.ceil(size[0]*factor).astype(int), 
                   np.ceil(size[1]*factor).astype(int))
    size = (np.ceil(size[0]).astype(int), 
            np.ceil(size[1]).astype(int))
    if is_too_wide:
        fontsize_mult = size[1]
        position = [position[0]*fontsize_mult+(size[0]-size[1]*aspect_ratio)/2,position[1]*fontsize_mult]
    elif is_too_tall:
        fontsize_mult = size[0]*(size[0]/size[1])/aspect_ratio
        position = [position[0]*fontsize_mult,position[1]*size[0]]
    else:
        fontsize_mult = size[1]
        position = [position[0]*fontsize_mult,position[1]*fontsize_mult]
    fontsize = np.floor(fontsize*fontsize_mult).astype(int)
    image = Image.new("L", size_factor, 0)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, 255, font=ImageFont.truetype(fontname, fontsize), anchor="mm")
    img_resized = image.resize(size, Image.Resampling.BOX)
    return img_resized

def pretty_point(im,footprint=None,radius=0.05):
    if footprint is None:
        #make star-shaped footprint
        min_sidelength = min(im.shape[0],im.shape[1])
        rad1 = np.ceil(radius*min_sidelength*0.66666).astype(int)
        rad2 = radius*min_sidelength*0.33333
        rad3 = np.ceil(rad1+rad2).astype(int)
        footprint = np.ones((2*rad3+1,2*rad3+1))
        #make cross
        footprint[rad3,rad3-rad1:rad3+rad1+1] = 0
        footprint[rad3-rad1:rad3+rad1+1,rad3] = 0
        footprint = nd.distance_transform_edt(footprint,return_indices=False)
        footprint = (footprint<=rad2).astype(int)
    else:
        assert isinstance(footprint,np.ndarray), "footprint must be a numpy array or None"
    if len(im.shape)==2:
        im = im[:,:,np.newaxis]
    if len(footprint.shape)==2:
        footprint = footprint[:,:,np.newaxis]
    #convolve image with footprint
    conv = nd.convolve(im,footprint,mode='constant',cval=0.0)
    conv_num = nd.convolve((np.abs(im)>1e-10).astype(float),footprint,mode='constant',cval=0.0)
    # Same as pretty_point_image = conv/conv_num, but avoiding 0/0
    pretty_point_image = conv
    pretty_point_image[conv_num>0] = conv[conv_num>0]/conv_num[conv_num>0]
    return pretty_point_image

def is_type_for_dot_shape(item):
    return isinstance(item,np.ndarray) or torch.is_tensor(item)

def is_deepest_expand(x,expand_deepest,max_expand):
    if expand_deepest:
        if isinstance(x,str):
            if len(x)<=max_expand:
                out = True
            else:
                out = False
        elif isinstance(x,(int,float)):
            out = True
        else:
            out = False
    else:
        out = False
    return out

def is_type_for_recursion(item,m=20):
    out = False
    if isinstance(item,(list,dict,tuple)):
        if len(item)<=m:
            out = True
    return out

def reduce_baseline(x):
    if hasattr(x,"__len__"):
        lenx = len(x)
    else:
        lenx = -1
    return f"<{type(x).__name__}>len{lenx}"

def fancy_shape(item):
    assert is_type_for_dot_shape(item)
    if torch.is_tensor(item):
        out = str(item.shape)
    else:
        out = f"np.Size({list(item.shape)})"
    return out

def shaprint(x, max_recursions=5, max_expand=20, first_only=False,do_pprint=True,do_print=False,return_str=False, expand_deepest=False):
    """
    Prints almost any object as a nested structure of shapes and lengths.
    Example:
    strange_object = {"a":np.random.rand(3,4,5),"b": [np.random.rand(3,4,5) for _ in range(3)],"c": {"d": [torch.rand(3,4,5),[[1,2,3],[4,5,6]]]}}
    shaprint(strange_object)
    """
    kwargs = {"max_recursions":max_recursions,
              "max_expand":max_expand,
              "first_only":first_only,
              "do_pprint": False,
              "do_print": False,
              "return_str": True,
              "expand_deepest":expand_deepest}
    m = float("inf") if first_only else max_expand
    if is_type_for_dot_shape(x):
        out = fancy_shape(x)
    elif is_deepest_expand(x,expand_deepest,max_expand):
        out = x
    elif is_type_for_recursion(x,m):
        if kwargs["max_recursions"]<=0:
            out = reduce_baseline(x)
        else:
            kwargs["max_recursions"] -= 1
            if isinstance(x,list):
                if first_only:
                    out = [shaprint(x[0],**kwargs)]
                else:
                    out = [shaprint(a,**kwargs) for a in x]
            elif isinstance(x,dict):
                if first_only:
                    k0 = list(x.keys())[0]
                    out = {k0: shaprint(x[k0],**kwargs)}
                else:
                    out = {k:shaprint(v,**kwargs) for k,v in x.items()}
            elif isinstance(x,tuple):
                if first_only:
                    out = tuple([shaprint(x[0],**kwargs)])
                else:
                    out = tuple([shaprint(a,**kwargs) for a in x])
    else:    
        out = reduce_baseline(x)
    if do_pprint:
        pprint(out)
    if do_print:
        print(out)
    if return_str:
        return out
        

class MatplotlibTempBackend():
    def __init__(self,backend):
        self.backend = backend
    def __enter__(self):
        self.old_backend = matplotlib.get_backend()
        matplotlib.use(self.backend)
    def __exit__(self, exc_type, exc_val, exc_tb):
        matplotlib.use(self.old_backend)

def quantile_normalize(x, alpha=0.001, q=None):
    if alpha is not None:
        assert q is None, "expected exactly 1 of alpha or q to be None"
        q = [alpha, 1-alpha]
    assert q is not None, "expected exactly 1 of alpha or q to be None"
    assert len(q)==2, "expected len(q)==2"
    minval,maxval = np.quantile(x,q)
    x = (x-minval)/(maxval-minval)
    x = np.clip(x,0,1)
    return x


def load_state_dict_loose(model_arch,state_dict,allow_diff_size=True,verbose=False):
    arch_state_dict = model_arch.state_dict()
    load_info = {"arch_not_sd": [],"sd_not_arch": [],"match_same_size": [], "match_diff_size": []}
    sd_keys = list(state_dict.keys())
    for name, W in arch_state_dict.items():
        if name in sd_keys:
            sd_keys.remove(name)
            s1 = np.array(state_dict[name].shape)
            s2 = np.array(W.shape)
            l1 = len(s1)
            l2 = len(s2)
            l_max = max(l1,l2)
            if l1<l_max:
                s1 = np.concatenate((s1,np.ones(l_max-l1,dtype=int)))
            if l2<l_max:
                s2 = np.concatenate((s2,np.ones(l_max-l2,dtype=int)))
                
            if all(s1==s2):
                load_info["match_same_size"].append(name)
                arch_state_dict[name] = state_dict[name]
            else:
                if verbose:
                    m = ". Matching." if allow_diff_size else ". Ignoring."
                    print("Param. "+name+" found with sizes: "+str(list(s1[0:l1]))
                                                      +" and "+str(list(s2[0:l2]))+m)
                if allow_diff_size:
                    s = [min(i_s1,i_s2) for i_s1,i_s2 in zip(list(s1),list(s2))]
                    idx1 = [slice(None,s[i],None) for i in range(l2)]
                    idx2 = tuple([slice(None,s[i],None) for i in range(l2)])
                    
                    if l1>l2:
                        idx1 += [0 for _ in range(l1-l2)]
                    idx1 = tuple(idx1)
                    tmp = state_dict[name][idx1]
                    arch_state_dict[name][idx2] = tmp
                load_info["match_diff_size"].append(name)
        else:
            load_info["arch_not_sd"].append(name)
    for name in sd_keys:
        load_info["sd_not_arch"].append(name)
    model_arch.load_state_dict(arch_state_dict)
    return model_arch, load_info

class TemporarilyDeterministic:
    def __init__(self,seed=0,torch=True,numpy=True):
        self.seed = seed
        self.torch = torch
        self.numpy = numpy
    def __enter__(self):
        if self.seed is not None:
            if self.numpy:
                self.previous_seed = np.random.get_state()[1][0]
                np.random.seed(self.seed)
            if self.torch:
                self.previous_torch_seed = torch.get_rng_state()
                torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is not None:
            if self.numpy:
                np.random.seed(self.previous_seed)
            if self.torch:
                torch.set_rng_state(self.previous_torch_seed)


class RenderMatplotlibAxis:
    def __init__(self, height, width=None, with_axis=False, set_lims=False, with_alpha=False, dpi=100):
        if (width is None) and isinstance(height, (tuple, list)):
            #height is a shape
            height,width = height[:2]
        elif (width is None) and isinstance(height, np.ndarray):
            #height is an image
            height,width = height.shape[:2]
        elif width is None:
            width = height
        self.with_alpha = with_alpha
        self.width = width
        self.height = height
        self.dpi = dpi
        self.old_backend = matplotlib.rcParams['backend']
        self.old_dpi = matplotlib.rcParams['figure.dpi']
        self.fig = None
        self.ax = None
        self._image = None
        self.with_axis = with_axis
        self.set_lims = set_lims

    @property
    def image(self):
        return self._image[:,:,:(3+int(self.with_alpha))]

    def __enter__(self):
        matplotlib.rcParams['figure.dpi'] = self.dpi
        matplotlib.use('Agg')
        figsize = (self.width/self.dpi, self.height/self.dpi)
        self.fig = plt.figure(figsize=figsize,dpi=self.dpi)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        if not self.with_axis:
            self.ax.set_frame_on(False)
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
        self.fig.add_axes(self.ax)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            # If no exception occurred, save the image to the _image property
            if self.set_lims:
                self.ax.set_xlim(-0.5, self.width-0.5)
                self.ax.set_ylim(self.height-0.5, -0.5)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', pad_inches=0, dpi=self.dpi)
            buf.seek(0)
            self._image = np.array(Image.open(buf))

        plt.close(self.fig)
        matplotlib.use(self.old_backend)
        matplotlib.rcParams['figure.dpi'] = self.old_dpi


def item_to_rect_lists(item,n1,n2,fill_with_previous=True, fill_val=None):
    fill_val0 = copy.copy(fill_val)
    if fill_with_previous:
        assert fill_val is None, "expected fill_val to be None if fill_with_previous is True" 
    if not isinstance(item,list):
        out = [[item]]
    else:
        if len(item)==0:
            out = [[[] for _ in range(n2)] for _ in range(n1)]
        else:
            if not isinstance(item[0],list):
                out = [item]
            else:
                out = item
    assert len(out)<=n1, f"expected len(out) to be <= {n1}, found {len(out)}"
    if len(out)<n1:
        out.extend([[] for _ in range(n1-len(out))])
    for i in range(len(out)):
        assert len(out[i])<=n2, f"expected len(out[{i}]) to be <= {n2}, found {len(out[i])}"
        if len(out[i])==0 and fill_with_previous:
            all_until_i = sum(out[:i],[])
            all = sum(out,[])
            if len(all_until_i)>0:
                out_i = all_until_i[-1]
            elif len(all)>0:
                out_i = all[-1]
            else:
                out_i = fill_val0
            out[i].append(out_i)
        if len(out[i])<n2:
            if fill_with_previous:
                fill_val = out[i][-1]
            out[i].extend([fill_val for _ in range(n2-len(out[i]))])
    return out


def render_text_gridlike(image, x_sizes, y_sizes, 
                        text_inside=[],
                        transpose_text_inside=False,
                        text_pos_kwargs={},
                        pixel_mult=1, 
                        text_kwargs={"color":"red","fontsize": 20,"verticalalignment":"bottom","horizontalalignment":"left"},
                        anchor_image="NW",
                        border_width_inside=0):
    nx = len(x_sizes)
    ny = len(y_sizes)
    anchor_image = item_to_rect_lists(copy.deepcopy(anchor_image),nx,ny)
    anchor_image = [[to_xy_anchor(a) for a in row] for row in anchor_image]

    if pixel_mult>1:
        h,w = image.shape[:2]
        h,w = (np.round(w*pixel_mult).astype(int),
               np.round(h*pixel_mult).astype(int))
        image = cv2.resize(copy.copy(image),(w,h))
    
    #make sure text_inside is a list of lists, with correct lengths
    text_inside = copy.deepcopy(text_inside)
    assert len(text_inside)<=nx, f"expected len(text_inside) to be <= len(x_sizes), found {len(text_inside)}>{nx}"
    for i in range(len(text_inside)):
        assert len(text_inside[i])<=ny, f"expected len(text_inside[{i}]) to be <= len(y_sizes), found {len(text_inside[i])}>{ny}"
    if len(text_inside)<nx:
        text_inside.extend([[] for _ in range(nx-len(text_inside))])
    for i in range(len(text_inside)):
        if len(text_inside[i])<ny:
            text_inside[i].extend(["" for _ in range(ny-len(text_inside[i]))])

    if transpose_text_inside:
        text_inside = list(zip(*text_inside))
    h,w = image.shape[:2]
    x_sum = sum(x_sizes)
    y_sum = sum(y_sizes)
    if not x_sum==1.0:
        x_sizes = [x/x_sum*w for x in x_sizes]
    if not y_sum==1.0:
        y_sizes = [y/y_sum*h for y in y_sizes]
    with RenderMatplotlibAxis(w,h,set_lims=1) as renderer: #TODO (is this an error? w,h should be switched)
        plt.imshow(image/255)
        for xi in range(len(x_sizes)):
            for yi in range(len(y_sizes)):
                anc_x,anc_y = anchor_image[xi][yi]
                x = sum(x_sizes[:xi])+anc_x*x_sizes[xi]
                y = sum(y_sizes[:yi])+anc_y*y_sizes[yi]
                if len(text_inside[xi][yi])>0:
                    txt = plt.text(x,y,text_inside[xi][yi],**text_kwargs)
                    if border_width_inside>0:
                        txt.set_path_effects([withStroke(linewidth=border_width_inside, foreground='black')])
    rendered = renderer.image
    valid_pos = ["top","bottom","left","right"]
    if any([k in text_pos_kwargs for k in valid_pos]):
        text_pos_kwargs2 = {"n_horz": len(x_sizes), "n_vert": len(y_sizes),"save": False, "buffer_pixels": 0, "add_spaces": 0}
        text_pos_kwargs2.update(text_pos_kwargs)
        rendered = add_text_axis_to_image(rendered,**text_pos_kwargs2)
    return rendered


def to_xy_anchor(anchor):
    anchor_equiv = [["NW","top left","north west","upper left","upper left corner"],
                    ["N","top","north","upper","top center","north center","upper center"],
                    ["NE","top right","north east","upper right","upper right corner"],
                    ["W","left","west","left center","west center","left middle","west middle","mid left"],
                    ["C","CC","center","middle","center middle", "middle center","center center","middle middle","mid"],
                    ["E","right","east","right center","east center","right middle","east middle","mid right"],
                    ["SW","bottom left","south west","lower left","lower left corner"],
                    ["S","bottom","south","lower","bottom center","south center","lower center"],
                    ["SE","bottom right","south east","lower right","lower right corner"]]
    anchor_to_coords = {"NW":(0,0),"N":(0.5,0),"NE":(1,0),
                        "W":(0,0.5),"C":(0.5,0.5),"E":(1,0.5),
                        "SW":(0,1),"S":(0.5,1),"SE":(1,1)}
    if isinstance(anchor,str):
        if anchor in sum(anchor_equiv,[]):
            for i,ae in enumerate(anchor_equiv):
                if anchor in ae:
                    out = anchor_to_coords[anchor_equiv[i][0]]
                    break
        else:
            raise ValueError(f"Unknown anchor string: {anchor}, Use on of {[x[0] for x in anchor_equiv]}")
    else:
        assert len(anchor)==2, f"If anchor is not a str then len(anchor) must be 2, found {len(anchor)}"
        out = anchor
    out = tuple([float(x) for x in out])
    return out

def add_text_axis_to_image(filename,
                           new_filename = None,
                           n_horz=None,n_vert=None,
                           top=[],bottom=[],left=[],right=[],
                           bg_color="white",
                           xtick_kwargs={},
                           new_file=False,
                           buffer_pixels=4,
                           add_spaces=True,
                           save=True):
    """
    Function to take an image filename and add text to the top, 
    bottom, left, and right of the image. The text is rendered
    using matplotlib and up to 4 temporary files are created to
    render the text. The temporary files are removed after the
    original file has been modified.

    Parameters
    ----------
    filename : str
        The filename of the image to modify.
    n_horz : int, optional
        The number of horizontal text labels to add. The default
        is None (max(len(top),len(bottom))).
    n_vert : int, optional
        The number of vertical text labels to add. The default
        is None (max(len(left),len(right))).
    top : list, optional
        The list of strings to add to the top of the image. The
        default is [].
    bottom : list, optional
        The list of strings to add to the bottom of the image. The
        default is [].
    left : list, optional
        The list of strings to add to the left of the image. The
        default is [].
    right : list, optional
        The list of strings to add to the right of the image. The
        default is [].
    bg_color : list, optional
        The background color of the text. The default is [1,1,1]
        (white).
    xtick_kwargs : dict, optional
        The keyword arguments to pass to matplotlib.pyplot.xticks.
        The default is {}.        
    new_file : bool, optional
        If True, then a new file is created with the text axis
        added. If False, then the original file is modified. The
        default is False.
    buffer_pixels : int, optional
        The number of pixels to add as a buffer between the image
        and the text. The default is 4.
    add_spaces : bool, optional
        If True, then a space is added to the beginning and end of
        each label. The default is True.
    save : bool, optional
        If True, then the new file is saved. The default is True.
        
    Returns
    -------
    im2 : np.ndarray
        The modified image with the text axis added.
    """
    if n_horz is None:
        n_horz = max(len(top),len(bottom))
    if n_vert is None:
        n_vert = max(len(left),len(right))
    if isinstance(filename,np.ndarray):
        im = filename
    else:
        assert os.path.exists(filename), f"filename {filename} does not exist"
        im = np.array(Image.open(filename))
    h,w,c = im.shape
    xtick_kwargs_per_pos = {"top":    {"rotation": 0,  "labels": top},
                            "bottom": {"rotation": 0,  "labels": bottom},
                            "left":   {"rotation": 90, "labels": left},
                            "right":  {"rotation": 90, "labels": right}}
    tick_params_per_pos = {"top":    {"top":True, "labeltop":True, "bottom":False, "labelbottom":False},
                           "bottom": {},
                           "left":   {},
                           "right":  {"top":True, "labeltop":True, "bottom":False, "labelbottom":False}}
    pos_renders = {}
    pos_sizes = {}
    for pos in ["top","bottom","left","right"]:
        if len(xtick_kwargs_per_pos[pos]["labels"])==0:
            pos_renders[pos] = np.zeros((0,0,c),dtype=np.uint8)
            pos_sizes[pos] = 0
            continue
        xk = dict(**xtick_kwargs_per_pos[pos],**xtick_kwargs)
        if add_spaces:
            xk["labels"] = [" "+l+" " for l in xk["labels"]]
        if not "ticks" in xk.keys():
            n = n_horz if pos in ["top","bottom"] else n_vert

            if len(xk["labels"])<n:
                xk["labels"] += [""]*(n-len(xk["labels"]))
            elif len(xk["labels"])>n:
                xk["labels"] = xk["labels"][:n]
            else:
                assert len(xk["labels"])==n
        pos_renders[pos] = render_axis_ticks(image_width=w if pos in ["top","bottom"] else h,
                                             num_uniform_spaced=n,
                                             bg_color=bg_color,
                                             xtick_kwargs=xk,
                                             tick_params=tick_params_per_pos[pos])[:,:,:c]
        pos_sizes[pos] = pos_renders[pos].shape[0]
    bg_color_3d = get_matplotlib_color(bg_color,c)
    bp = buffer_pixels
    im2 = np.zeros((h+pos_sizes["top"]+pos_sizes["bottom"]+bp*2,
                    w+pos_sizes["left"]+pos_sizes["right"]+bp*2,
                    c),dtype=np.uint8)
    im2 += bg_color_3d
    im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,
        bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = im
    #make sure we have uint8
    pos_renders = {k: np.clip(v,0,255) for k,v in pos_renders.items()}
    for pos in ["top","bottom","left","right"]:
        if pos_renders[pos].size==0:
            continue
        if pos=="top":
            im2[bp:bp+pos_sizes["top"],bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["top"]
        elif pos=="bottom":
            im2[bp+pos_sizes["top"]+h:-bp,bp+pos_sizes["left"]:bp+pos_sizes["left"]+w] = pos_renders["bottom"]
        elif pos=="left":
            im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,bp:bp+pos_sizes["left"]] = np.rot90(pos_renders["left"],k=3)
        elif pos=="right":
            im2[bp+pos_sizes["top"]:bp+pos_sizes["top"]+h,bp+pos_sizes["left"]+w:-bp] = np.rot90(pos_renders["right"],k=3)
    if new_file:
        if new_filename is None:
            suffix = filename.split(".")[-1]
            new_filename = filename[:-len(suffix)-1]+"_w_text."+suffix
            for i in range(1000):
                if not os.path.exists(new_filename):
                    break
                new_filename = filename[:-len(suffix)-1]+"_w_text("+str(i)+")."+suffix
        filename = new_filename
    if save:
        Image.fromarray(im2).save(filename)
    return im2

def get_matplotlib_color(color,num_channels=3):
    return render_axis_ticks(23,bg_color=color,xtick_kwargs={"labels": [" "]}, tick_params={"bottom": False})[12,12,:num_channels]

def darker_color(x,power=2,mult=0.5):
    assert isinstance(x,np.ndarray), "darker_color expects an np.ndarray"
    is_int_type = x.dtype in [np.uint8,np.uint16,np.int8,np.int16,np.int32,np.int64]
    if is_int_type:
        return np.round(255*darker_color(x/255,power=power,mult=mult)).astype(np.uint8)
    else:
        return np.clip(x**power*mult,0,1)

def render_axis_ticks(image_width=1000,
                      num_uniform_spaced=None,
                      bg_color="white",
                      xtick_kwargs={"labels": np.arange(5)},
                      tick_params={}):
    old_backend = matplotlib.rcParams['backend']
    old_dpi = matplotlib.rcParams['figure.dpi']
    dpi = 100
    if num_uniform_spaced is None:
        num_uniform_spaced = len(xtick_kwargs["labels"])
    n = num_uniform_spaced
     
    matplotlib.rcParams['figure.dpi'] = dpi
    matplotlib.use('Agg')
    try:        
        fig = plt.figure(figsize=(image_width/dpi, 1e-15), facecolor=bg_color)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_facecolor(bg_color)
        ax.set_frame_on(False)
        ax.tick_params(**tick_params)
        fig.add_axes(ax)
        
        plt.yticks([])
        plt.xlim(0, n)
        x_pos = np.linspace(0.5,n-0.5,n)
        if not "ticks" in xtick_kwargs:
            xtick_kwargs["ticks"] = x_pos[:len(xtick_kwargs["labels"])]
        else:
            if xtick_kwargs["ticks"] is None:
                xtick_kwargs["ticks"] = x_pos[:len(xtick_kwargs["labels"])]
        plt.xticks(**xtick_kwargs)
        
        with warnings.catch_warnings(record=True) as caught_warnings:
            fig.show()

        with NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_filename = temp_file.name
            fig.savefig(temp_filename, format='png', bbox_inches='tight', pad_inches=0)
        im = np.array(Image.open(temp_filename))
        if not im.shape[1]==image_width:
            #reshape with cv2 linear interpolation
            #warnings.warn("Image width is not as expected, likely due to too large text labels. Reshaping with cv2 linear interpolation.")
            im = cv2.resize(im, (image_width, im.shape[0]), interpolation=cv2.INTER_LINEAR)

        matplotlib.use(old_backend)
        matplotlib.rcParams['figure.dpi'] = old_dpi
    except:
        matplotlib.use(old_backend)
        matplotlib.rcParams['figure.dpi'] = old_dpi
        raise
    return im

def get_mask(mask_vol,idx,onehot=False,onehot_dim=-1):
    if onehot:
        slice_idx = [slice(None) for _ in range(len(mask_vol.shape))]
        slice_idx[onehot_dim] = idx
        return np.expand_dims(mask_vol[tuple(slice_idx)],onehot_dim)
    else:
        return (mask_vol==idx).astype(float)

def mask_overlay_smooth(image,
                        mask,
                        num_spatial_dims=2,
                        pallete=None,
                        pixel_mult=1,
                        class_names=None,
                        show_border=False,
                        border_color="darker",
                        alpha_mask=0.4,
                        dont_show_idx=[255],
                        fontsize=12,
                        text_color="class",
                        text_alpha=1.0,
                        text_border_instead_of_background=True,
                        set_lims=True):
    assert isinstance(image,np.ndarray)
    assert isinstance(mask,np.ndarray)
    assert len(image.shape)>=num_spatial_dims, "image must have at least num_spatial_dims dimensions"
    assert len(mask.shape)>=num_spatial_dims, "mask must have at least num_spatial_dims dimensions"
    assert image.shape[:num_spatial_dims]==mask.shape[:num_spatial_dims], "image and mask must have the same shape"
    if pallete is None:
        pallete = np.concatenate([np.array([[0,0,0]]),nc.largest_colors],axis=0)
    if image.dtype==np.uint8:
        was_uint8 = True
        image = image.astype(float)/255
    else:
        was_uint8 = False
    if len(mask.shape)==num_spatial_dims:
        onehot = False
        n = mask.max()+1
        uq = np.unique(mask).tolist()
        mask = np.expand_dims(mask,-1)
    else:
        assert len(mask.shape)==num_spatial_dims+1, "mask must have num_spatial_dims (with integers as classes) or num_spatial_dims+1 dimensions (with onehot encoding)"
        if mask.shape[num_spatial_dims]==1:
            onehot = False
            n = mask.max()+1
            uq = np.unique(mask).tolist()
        else:
            onehot = True
            n = mask.shape[num_spatial_dims]
            uq = np.arange(n).tolist()
    image_colored = image.copy()
    if len(image_colored.shape)==num_spatial_dims:
        image_colored = np.expand_dims(image_colored,-1)
    #make rgb
    if image_colored.shape[-1]==1:
        image_colored = np.repeat(image_colored,3,axis=-1)
    color_shape = tuple([1 for _ in range(num_spatial_dims)])+(3,)
    show_idx = [i for i in uq if (not i in dont_show_idx)]
    for i in show_idx:
        reshaped_color = pallete[i].reshape(color_shape)/255
        mask_coef = alpha_mask*get_mask(mask,i,onehot=onehot)
        image_coef = 1-mask_coef
        image_colored = image_colored*image_coef+reshaped_color*mask_coef
    if class_names is not None:
        assert isinstance(class_names,dict), "class_names must be a dictionary that maps class indices to class names"
        for i in uq:
            assert i in class_names.keys(), f"class_names must have a key for each class index, found i={i} not in class_names.keys()"
    assert isinstance(pixel_mult,int), "pixel_mult must be an integer"
    
    if pixel_mult>1:
        image_colored = cv2.resize(image_colored,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_NEAREST)
    
    image_colored = np.clip(image_colored,0,1)
    if show_border or (class_names is not None):
        image_colored = (image_colored*255).astype(np.uint8)
        h,w = image_colored.shape[:2]
        with RenderMatplotlibAxis(h,w,set_lims=set_lims) as ax:
            plt.imshow(image_colored)
            for i in show_idx:
                mask_coef = get_mask(mask,i,onehot=onehot)
                if pixel_mult>1:
                    mask_coef = cv2.resize(mask_coef,None,fx=pixel_mult,fy=pixel_mult,interpolation=cv2.INTER_LANCZOS4)
                else:
                    mask_coef = mask_coef.reshape(h,w)
                if show_border:                    
                    curves = find_contours(mask_coef, 0.5)
                    if border_color=="darker":
                        border_color_i = darker_color(pallete[i]/255)
                    else:
                        border_color_i = border_color
                    k = 0
                    for curve in curves:
                        plt.plot(curve[:, 1], curve[:, 0], linewidth=1, color=border_color_i)
                        k += 1

                if class_names is not None:
                    t = class_names[i]
                    if len(t)>0:
                        dist = distance_transform_edt_border(mask_coef)
                        y,x = np.unravel_index(np.argmax(dist),dist.shape)
                        if text_color=="class":
                            text_color_i = pallete[i]/255
                        else:
                            text_color_i = text_color
                        text_kwargs = {"fontsize": int(fontsize*pixel_mult),
                                       "color": text_color_i,
                                       "alpha": text_alpha}
                        col_bg = "black" if np.mean(text_color_i)>0.5 else "white"             
                        t = plt.text(x,y,t,**text_kwargs)
                        if text_border_instead_of_background:
                            t.set_path_effects([withStroke(linewidth=3, foreground=col_bg)])
                        else:
                            t.set_bbox(dict(facecolor=col_bg, alpha=text_alpha, linewidth=0))
        image_colored = ax.image
    else:
        if was_uint8: 
            image_colored = (image_colored*255).astype(np.uint8)
    return image_colored

def distance_transform_edt_border(mask):
    padded = np.pad(mask,1,mode="constant",constant_values=0)
    dist = nd.distance_transform_edt(padded)
    return dist[1:-1,1:-1]