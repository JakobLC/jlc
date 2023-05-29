import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import PIL
import torch
import cv2
import copy 
import random

def montage(arr,
            maintain_aspect=True,
            reshape=True,
            imshow=True,
            reshape_size=None,
            n_col=None,
            n_row=None,
            padding=0,
            padding_color=0,
            rows_first=True,
            figsize_per_pixel=1/100):
    """
    Function that displays and returns an montage of images from a list or 
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
    Returns
    -------
    im_cat : np.array
        Concatenated montage image.
        
    
    Example
    -------
    montage(np.random.rand(2,3,4,5,3),reshape_size=(40,50))

    """
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
    
    channels = 1
    for n,i,j in zip(N,I,J): 
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
    im_cat = np.clip(im_cat,0,1)
    if imshow:
        plt.figure(figsize=(figsize_per_pixel*im_cat.shape[1],figsize_per_pixel*im_cat.shape[0]))
        plt.imshow(im_cat,cmap="gray")
        plt.show()
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
    else:
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