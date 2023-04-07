import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
#from PIL import Image

def montage(arr,
            maintain_aspect=True,
            reshape=True,
            imshow=True,
            reshape_size=None,
            n_col=None,
            n_row=None,
            padding=0,
            padding_color=0,
            rows_first=True):
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
                        #print((i+1)*n1+j-1)
                        arr2[i].append(arr[ii])
        else:
            arr2 = [[] for _ in range(n1)]
            for j in range(n2):
                for i in range(n1):
                    ii = i+j*n2
                    if ii<len(arr):
                        #print((i+1)*n1+j-1)
                        arr2[i].append(arr[ii])
        arr = arr2
        
    n1 = len(arr)
    n2_list = [len(arr[i]) for i in range(n1)]
    if n_row is None:
        n2 = max(n2_list)
    else:
        n2 = n_row
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
        im_c = 1
        if len(im.shape)>2:
            im_c = im.shape[2]
        else:
            im = im[:,:,None]
            
        if im_c<channels:
            if channels>=3 and im_c==1:
                if len(im.shape)>2:
                    im = im[:,:,0]
                im = np.stack([im]*3,axis=2)
            if channels==4 and im_c<4:
                im = np.concatenate([im]+[np.ones((d1,d2,1))],axis=2)
        im_cat[idx_d1,idx_d2,:] = im
    im_cat = np.clip(im_cat,0,1)
    if imshow:
        plt.imshow(im_cat)
    return im_cat