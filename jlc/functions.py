import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import PIL
import torch
import cv2
import copy 
import random
from PIL import Image, ImageDraw, ImageFont

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