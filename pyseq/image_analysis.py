#!/usr/bin/python

import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import xarray as xr
xr.set_options(keep_attrs=True)
import zarr
import napari
import pandas as pd
from math import log2, ceil, floor
from os import listdir, stat, path, getcwd, mkdir
from scipy import stats
from scipy.spatial.distance import cdist
from skimage.exposure import match_histograms
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
import imageio
import glob
import configparser
import time
from qtpy.QtCore import QTimer

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from . import resources
from . import recipes

def message(logger, *args):
    """Print output text to logger or console.

       If there is no logger, text is printed to the console.
       If a logger is assigned, text is printed to the log file.

    """

    msg = 'ImageAnalysis::'
    for a in args:
        msg += str(a) + ' '

    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def get_image_df(image_path = None, image_name = None, im_type = '.tiff'):
    """Get dataframe of rough focus images.

       **Parameters:**
       image_path (path): Directory where images are stored.
       image_name (str): Name common to all images.
       im_type = (str): Image type suffix common to all images

       **Returns:**
       dataframe: Dataframe of image metadata with image names as index.

    """

    if image_path is None:
        image_path = getcwd()

    all_names = listdir(image_path)
    if image_name is None:
      image_names = [name for name in all_names if im_type in name]
    else:
      im_names = [name for name in all_names if image_name in name]
      image_names = [name for name in im_names if im_type in name]

    # Dataframe for metdata
    cols = image_names[0][:-5].split('_')
    col_int = []
    col_names = []
    i = 0
    for c in cols:
    # Get column names and test if it should be an integer
        try:
            int(c[0])
            col_names.append('X' + str(i))
            col_int.append(1)
        except:
            if c in col_names:
                col_names.append(c[0]+str(i))
            else:
                col_names.append(c[0])

            try:
                int(c[1:])
                col_int.append(2)
            except:
                col_int.append(0)

        i += 1

    metadata = pd.DataFrame(columns = (col_names))

    # Extract metadata
    for name in image_names:

      meta = name[:-5].split('_')
      for i in range(len(col_int)):
          if col_int[i]:
              try:
                  meta[i] = int(meta[i][col_int[i]-1:])
              except:
                  print(name)

      metadata.loc[name] = meta

    return metadata

def stitch(dir, df_x, overlap = 0, scaled = False):
    """Stitch together scans.

       # TODO: Implement overlap

       **Parameters:**
       dir (path): Directory where image scans are stored.
       df_x (dataframe): Dataframe of metadata of image scans to stitch.
       overlap (int): Number of pixels that overlap between adjacent tiles.
       scaled (bool): True to autoscale images to ~256 Kb, False to not scale.

       **Returns:**
       array: Stitched and scaled image.
       int: Downscale factor the images were scaled by.

    """

    df_x = df_x.sort_values(by=['x'])
    scale_factor = 1
    plane = None
    for name in df_x.index:
        im = imageio.imread(path.join(dir,name))
        im = im[64:]                                                            # Remove whiteband artifact

        if scaled:
            # Scale images so they are ~ 512 kb
            if scale_factor == 1:
                size = stat(path.join(dir,name)).st_size
                scale_factor = (2**(log2(size)-19))**0.5
                scale_factor = round(log2(scale_factor))
                scale_factor = 2**scale_factor

            im = downscale_local_mean(im, (scale_factor, scale_factor))

        # Append scans together
        if plane is None:
            plane = im
        else:
            plane = np.append(plane, im, axis = 1)

    plane = plane.astype('uint16')

    return plane, scale_factor

def normalize(im, scale_factor):
    """Normalize scans.

       Images are normalized by matching histogram to strip with most contrast.


       **Parameters:**
       im (array): Image as array.
       scale_factor (int): Downscale factor of image.

       **Returns:**
       image: Normalized image.

    """

    x_px = int(2048/scale_factor/8)
    n_strips = int(im.shape[1]/x_px)
    strip_contrast = np.empty(shape=(n_strips,))
    #col_contrast = np.max(im, axis = 0) - np.min(im, axis = 0)

    # Find reference strip with max contrast
    for i in range(n_strips):
        strip_contrast[i] = np.std(im[:,(i)*x_px:(i+1)*x_px])
    ref_strip = np.argmax(strip_contrast)
    ref = im[:,(ref_strip)*x_px:(ref_strip+1)*x_px]

    plane = None
    for i in range(n_strips):
      sub_im = im[:,(i)*x_px:(i+1)*x_px]
      sub_im = match_histograms(sub_im, ref)
      if plane is None:
        plane = sub_im
      else:
        plane = np.append(plane, sub_im, axis = 1)

    plane = plane.astype('uint16')
    #plane = img_as_ubyte(plane)

    return plane


def sum_images(images, logger = None):
    """Sum pixel values over channel images.

       The image with the largest signal to noise ratio is used as the
       reference. Images without significant positive kurtosis, ie pixels that
       deviate from mean value (assumed as background), are discarded. The
       remaining image histograms are matched to the reference. Finally, the
       images are summed together. The summed image is returned or if there is
       no signal in all channels, False is returned.

       Parameters:
       - images (list): List of images to sum.
       - thresh (float): Kurtosis threshold to call signal in channel.
       - logger (logger): Logger object to record process.

       Return:
       - array: Numpy array of summed image or False if no signal

    """

    name_ = 'SumImages:'
    sum_im = None
    finished = False


    thresh = 81.0
    i = 0
    while not finished:
        message(logger, name_, 'kurtosis threshold (k) = ', thresh)
        for c, im in enumerate(images):
            #kurt_z, pvalue = stats.kurtosistest(im, axis = None)
            #kurt_z = stats.kurtosis(im, axis=None)
            k = kurt(im)
            message(logger, name_, 'Channel',c, 'k = ', k)
            if k > thresh:
                # Add add image
                if sum_im is None:
                    sum_im = im.astype('int16')
                    finished = True
                else:
                    sum_im = np.add(sum_im, im)

        if sum_im is None:
            if i <= 3:
                thresh = 3**(3-i)
                i+=1
            else:
                finished = True

    return sum_im

def sum_images2(images, logger = None):
    """Sum pixel values over channel images.

       The image with the largest signal to noise ratio is used as the
       reference. Images without significant positive kurtosis, ie pixels that
       deviate from mean value (assumed as background), are discarded. The
       remaining image histograms are matched to the reference. Finally, the
       images are summed together. The summed image is returned or if there is
       no signal in all channels, False is returned.

       Parameters:
       - images (data array): Xarray data array of images
       - logger (logger): Logger object to record process.

       Return:
       - array: Xarray data array of summed image or None if no signal

    """

    name_ = 'SumImages:'
    sum_im = None
    thresh = [1, 9, 27, 81]


    # Calculate modified kurtosis
    channels = images.channel.values
    k_dict = {}
    for ch in channels:
        k = kurt2(images.sel(channel=ch))
        message(logger, name_, 'Channel',ch, 'k = ', k)
        k_dict[ch] = k

    # Pick kurtosis threshold
    max_k = max(list(k_dict.values()))
    thresh_ind = np.where(max_k>np.array(thresh))[0]
    if len(thresh_ind) > 0:
        thresh = thresh[max(thresh_ind)]
        message(logger, name_, 'kurtosis threshold (k) = ', thresh)

        # keep channels with high kurtosis
        keep_ch = [ch for ch in channels if k_dict[ch] > thresh]
        im = images.sel(channel = keep_ch)

        # Sum remaining channels
        im = im.sum(dim='channel')
    else:
        im = None

    return im

def kurt(im):
    """Return kurtosis = mean((image-mode)/2)^4). """

    im = im.astype('int16')
    mode = stats.mode(im, axis = None)[0][0]
    z_score = (im-mode)/2
    k = np.mean(z_score**4)

    return k

def kurt2(im):
    """Return kurtosis = mean((image-mode)/2)^4). """

    mode = stats.mode(im, axis = None)[0][0]
    z_score = (im-mode)/2
    z_score = z_score**4
    k = float(z_score.mean().values)

    return k

def get_focus_points(im, scale, min_n_markers, log=None, p_sat = 99.9):
    """Get potential points to focus on.

       First 1000 of the brightest, unsaturated pixels are found.
       Then the focus field of views with the top *min_n_markers* contrast are
       ordered based on the distance from each other, with the points farthest
       away from each other being first.

       **Parameters:**
       - im (array): Summed image across all channels with signal.
       - scale (int): Factor at which the image is scaled down.
       - min_n_markers (int): Minimun number of points desired, max is 1000.
       - p_sat (float): Percentile to call pixels saturated.
       - log (logger): Logger object to record process.


       **Returns:**
       - array: Row, Column list of ordered pixels to use as focus points.

    """

    name_ = 'GetFocusPoints::'
    #score pixels
    px_rows, px_cols = im.shape
    px_sat = np.percentile(im, p_sat)
    px_score = np.reshape(stats.zscore(im, axis = None), (px_rows, px_cols))

    # Find brightest unsaturated pixels
    edge_width = int(2048/scale/2)
    im_ = np.zeros_like(im)
    px_score_thresh = 3
    while np.sum(im_ != 0) < min_n_markers:
        # Get brightest pixels
        im_[px_score > px_score_thresh] = im[px_score > px_score_thresh]
        # Remove "saturated" pixels
        im_[im > px_sat] = 0
        #Remove Edges
        if edge_width < px_cols/2:
          im_[:, px_cols-edge_width:px_cols] = 0
          im_[:,0:edge_width] = 0
        if edge_width < px_rows/2:
          im_[0:edge_width,:] = 0
          im_[px_rows-edge_width:px_rows, :] = 0

        px_score_thresh -= 0.5


    px_score_thresh += 0.5
    message(log, name_, 'Used', px_score_thresh, 'pixel score threshold')
    markers = np.argwhere(im_ != 0)


    # Subset to 1000 points
    n_markers = len(markers)
    message(log, name_, 'Found', n_markers, 'markers')
    if n_markers > 1000:
      rand_markers = np.random.choice(range(n_markers), size = 1000)
      markers = markers[rand_markers,:]
      n_markers = 1000

    # Compute contrast
    c_score = np.zeros_like(markers[:,1])
    for row in range(n_markers):
      mark = markers[row,:]
      if edge_width < px_cols/2:
        frame = im[mark[0],mark[1]-edge_width:mark[1]+edge_width]
      else:
        frame = im[mark[0],:]
      c_score[row] = np.max(frame) - np.min(frame)


    # Get the minimum number of markers needed with the highest contrast
    if n_markers > min_n_markers:
      p_top = (1 - min_n_markers/n_markers)*100
    else:
      p_top = 0
    message(log, name_, 'Used', p_top, 'percentile cutoff')
    c_cutoff = np.percentile(c_score, p_top)
    c_markers = markers[c_score >= c_cutoff,:]

    # Compute distance matrix
    dist = cdist(c_markers, c_markers)
    max_ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)          #returns tuple

    # Order marker points based on distance from each other
    # Markers farthest from each other are first
    n_markers = len(c_markers)
    ord_points = np.zeros_like(c_markers)
    if n_markers > 2:
        ord_points[0,:] = c_markers[max_ind[0],:]
        ord_points[1,:] = c_markers[max_ind[1],:]
        _markers = np.copy(c_markers)
        prev2 = max_ind[0]
        prev1 = max_ind[1]
        dist = np.delete(dist,[prev2,prev1],1)
        _markers = np.delete(_markers,[prev2,prev1], axis=0)
        for i in range(2,n_markers):
          dist2 = np.array([dist[prev2,:],dist[prev1,:]])
          ind = np.argmax(np.sum(dist2,axis=0))
          ord_points[i,:] = _markers[ind,:]
          dist = np.delete(dist,ind,1)
          _markers = np.delete(_markers,ind, axis=0)
          prev2 = prev1
          prev1 = ind
    else:
        ord_points = c_markers

    return ord_points

def get_focus_points_partial(im, scale, min_n_markers, log=None, p_sat = 99.9):
    """Get potential points to focus on.

       First 1000 of the brightest, unsaturated pixels are found.
       Then the focus field of views with the top *min_n_markers* contrast are
       ordered based on the distance from each other, with the points farthest
       away from each other being first.

       **Parameters:**
       - im (array): Summed image across all channels with signal.
       - scale (int): Factor at which the image is scaled down.
       - min_n_markers (int): Minimun number of points desired, max is 1000.
       - p_sat (float): Percentile to call pixels saturated.
       - log (logger): Logger object to record process.


       **Returns:**
       - array: Row, Column list of ordered pixels to use as focus points.

    """

    name_ = 'GetFocusPointsPartial::'
    #score pixels
    px_rows, px_cols = im.shape
    px_sat = np.percentile(im, p_sat)
    px_score = np.reshape(stats.zscore(im, axis = None), (px_rows, px_cols))

    # Find brightest unsaturated pixels
    edge_width = int(2048/scale/2)
    im_ = np.zeros_like(im)
    px_score_thresh = 3
    while np.sum(im_ != 0) < min_n_markers:
        # Get brightest pixels
        im_[px_score > px_score_thresh] = im[px_score > px_score_thresh]
        # Remove "saturated" pixels
        im_[im > px_sat] = 0
        #Remove Edges
        if edge_width < px_cols/2:
          im_[:, px_cols-edge_width:px_cols] = 0
          im_[:,0:edge_width] = 0
        if edge_width < px_rows/2:
          im_[0:edge_width,:] = 0
          im_[px_rows-edge_width:px_rows, :] = 0

        px_score_thresh -= 0.5


    px_score_thresh += 0.5
    message(log, name_, 'Used', px_score_thresh, 'pixel score threshold')
    markers = np.argwhere(im_ != 0)

    #Subset to unique y positions
    markers = np.array([*set(markers[:,0])])

    # Subset to 1000 points
    n_markers = len(markers)
    message(log, name_, 'Found', n_markers, 'markers')
    if n_markers > 1000:
      rand_markers = np.random.choice(range(n_markers), size = 1000)
      markers = markers[rand_markers,:]
      n_markers = 1000



    # Compute contrast
    c_score = np.zeros_like(markers)
    for row in range(n_markers):
      frame = im[markers[row],:]
      c_score[row] = np.max(frame) - np.min(frame)

    # Get the minimum number of markers needed with the highest contrast
    if n_markers > min_n_markers:
      p_top = (1 - min_n_markers/n_markers)*100
    else:
      p_top = 0
    message(log, name_, 'Used', p_top, 'percentile cutoff')
    c_cutoff = np.percentile(c_score, p_top)
    c_markers = markers[c_score >= c_cutoff]

    n_markers = len(c_markers)
    dist = np.ones((n_markers,n_markers))*c_markers
    dist = abs(dist-dist.T)
    # Compute distance matrix
    max_ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)          #returns tuple

    # Order marker points based on distance from each other
    # Markers farthest from each other are first
    ord_points = np.zeros_like(c_markers)
    if n_markers > 2:
        ord_points[0] = c_markers[max_ind[0]]
        ord_points[1] = c_markers[max_ind[1]]
        _markers = np.copy(c_markers)
        prev2 = max_ind[0]
        prev1 = max_ind[1]
        dist = np.delete(dist,[prev2,prev1], 1)
        _markers = np.delete(_markers,[prev2,prev1])
        for i in range(2,n_markers):
          dist2 = np.array([dist[prev2,:],dist[prev1,:]])
          ind = np.argmax(np.sum(dist2,axis=0))
          ord_points[i] = _markers[ind]
          dist = np.delete(dist,ind,1)
          _markers = np.delete(_markers,ind)
          prev2 = prev1
          prev1 = ind
    else:
        ord_points = c_markers

    ord_points = np.array([c_markers,np.ones(n_markers)*edge_width]).T

    return ord_points

def make_image(im_path, df_x, comp=None):
    """Make full scale, high quality, images.

       Stitch and normalize images in *df_x*. If using color compensation
       *comp*, include the linear model of the signal crosstalk from a lower
       wavelength channel to a higher wavelength channel as such
       {lower channel (nm, int): {'m': slope (float), 'b': constant (float),
       upper channel (nm, int)}}.

       # TODO: Implement overlap

       **Parameters:**
       im_path (path): Directory where image scans are stored.
       df_x (dataframe): Dataframe of metadata of image scans to stitch.
       comp (dict): Color compensation dictionary, optional.

       **Returns:**
       array: Stitched and scaled image.

    """


    df_x = df_x.sort_values(by=['x'])
    x_px = int(2048/8)
    n_strips = len(df_x)*8
    strip_contrast = np.empty(shape=(n_strips,))
    ref_strip_ = np.zeros_like(strip_contrast)
    ref_x_ = np.zeros_like(strip_contrast)

    # Strip with most constrast is reference
    for x, name in enumerate(df_x.index):
        im = imageio.imread(path.join(im_path,name))
        im = im[64:]                                                            # Remove whiteband artifact

        col_contrast = np.max(im, axis = 0) - np.min(im, axis = 0)
        for i in range(8):
            strip_contrast[i+x] = np.mean(col_contrast[(i)*x_px:(i+1)*x_px])
            ref_strip_[i+x] = i
            ref_x_[i+x] = x

    ref_strip = np.argmax(strip_contrast)
    ref_x = ref_x_[ref_strip]
    ref_strip = ref_strip_[ref_strip]

    name = df_x.index[ref_x]
    print('Reference is image:', name, 'strip', ref_strip)
    im = imageio.imread(path.join(im_path,name))
    ref_start = int(ref_strip*x_px)
    ref_stop = int((ref_strip+1)*x_px)
    ref = im[:,ref_start:ref_stop]

    # stitch images
    plane = None
    for name, row in df_x.iterrows():
        print('Stitching', name)
        im = imageio.imread(path.join(im_path,name))
        im = im[64:]                                                            # Remove whiteband artifact

        # color compensate
        if comp:
              if comp[row.c]:
                  c = comp[row.c]['c']
                  m = comp[row.c]['m']
                  b = comp[row.c]['b']
                  c_name = 'c' + str(c) + name[4:]
                  c_im = imageio.imread(path.join(im_path,c_name))
                  c_im = c_im[64:]
                  bleed = m*c_im+b                                                  # Calculate bleed over
                  #bleed = bleed.astype('int8')
                  bleed = bleed.astype('int16')
                  bg = np.zeros_like(c_im)
                  for c_i in range(8):
                      c_start = c_i*x_px
                      c_stop = c_start+x_px
                      bg[:,c_start:c_stop] = np.mean(c_im[:,c_start:c_stop])



        for i in range(8):
          if comp:
              if comp[row.c]:
                  sub_bleed = bleed[:,i*x_px:(i+1)*x_px]
                  sub_bg = bg[:,i*x_px:(i+1)*x_px]
                  sub_im = im[:,i*x_px:(i+1)*x_px]
                  sub_im = sub_im - sub_bleed + sub_bg
              else:
                  sub_im = im[:,i*x_px:(i+1)*x_px]
          else:
              sub_im = im[:,i*x_px:(i+1)*x_px]


          sub_im = match_histograms(sub_im, ref)
          if plane is None:
            #plane = sub_im.astype('int8')
            plane = sub_im.astype('int16')
          else:
            #plane = np.append(plane, sub_im.astype('int8'), axis = 1)
            plane = np.append(plane, sub_im.astype('int16'), axis = 1)


    #plane = plane.astype('int8')

    return plane

def tiff2zarr(image_path):
    """Convert HiSeq tiffs to zarr.

       **Parameters:**
       - im_path(path): Path to directory with tiff images

       **Returns:**
       - path: Path to zarr images

    """

    # Make tiffs into zarr
    base_dir = path.dirname(image_path)
    zarr_dir = path.join(base_dir, 'zarr')

    df = ia.get_image_df(tiff_dir)

    if len(df) > 0:
        col_names = ['s','r','c','o','x']
        extra_col = [col for col in df.columns if col not in col_names]
        col_names = [col for col in col_names if col not in df.columns]

        if len(df.columns) > 6:
            print('Extra columns:', *extra_col)
        elif len(col_name) == 0:
            sections = set(df.s)
            channels = set(df.c)
            rounds = set(df.r)

            for s in sections:
                df_s = df[df.s == s]
                (nrows, ncols) = imageio.imread(path.join(tiff_dir,df_s.index[0])).shape
                obj_steps = [*set(df_s.o)]
                obj_steps.sort()
                nrows -= 64
                ncols = 2048*len(set(df_s.x))
                for r in rounds:
                    df_r = df_s[df_s.r == r]
                    for c in channels:
                        df_c = df_r[df_r.c == c]
                        print('Converting section', s, 'channel', c, 'round', r)
                        for o in obj_steps:
                            df_o = df_c[df_c.o == o]
                            df_x = df_o.sort_values(by='x')
                            stack = []
                            for index, row in df_x.iterrows():
                                im = da.delayed(imageio.imread(path.join(tiff_dir,index)))
                                im = im[64:,]
                                stack.append(da.array(im))
                            im = da.concatenate(stack, axis=-1)
                            im = zarr.create(im, chunks = (nrows, 256), dtype='int16')
                            zarr_name = 'c'+str(c)+'_s'+str(s)+'_r'+str(r)+'_o'+str(o)+'.zarr'
                            zarr.save(path.join(zarr_dir, zarr_name), im)

        else:
            print('Missing', *col, 'columns')

    return zarr_dir

def label_images(df, zarr_path):
    """Label image dataset with section name, channel, cycle, and objective step.

       **Parameters:**
       - df(dataframe): Dataframe with image metadata

       **Returns:**
       - array: Labeled dataset

    """

    sections = [s[1:] for s in [*set(df.s)]]
    channels = [*set(df.c)]
    obj_steps = [*set(df.o)]
    obj_steps.sort()
    cycles = [*set(df.r)]
    cycles.sort()

    dim_names = ['section','channel', 'cycle', 'obj_step', 'row', 'col']
    coord_values = {'section':sections,'channel':channels,
                    'obj_step':obj_steps, 'cycle':cycles}

    s_stack = []
    for s in sections:
        c_stack = []
        for c in channels:
            df_c = df[df.c==c]
            r_stack = []
            for r in cycles:
                df_r = df_c[df_c.r==r]
                o_stack = []
                for o in obj_steps:
                    im_ = df_r[df_r.o==o]
                    if len(im_) == 1:
                        im_name = df_r[df_r.o==o].index[0]
                        o_stack.append(da.from_zarr(path.join(zarr_path,im_name)))
                r_stack.append(da.stack(o_stack,axis = 0))
            c_stack.append(da.stack(r_stack,axis = 0))
        s_stack.append(da.stack(c_stack,axis = 0))
    dataset = da.stack(s_stack, axis=0)
    dataset = xr.DataArray(dataset, dims = dim_names,coords=coord_values)

    return dataset.squeeze()


def compute_background(im_path, save_path = None, machine=''):

    ims = HiSeqImages(im_path)
    name = [*ims.sections.keys()][0]
    dataset = ims.sections[name]
    sensor_size = 256 # pixels
    background = {}

    # Loop over channels then sensor group and find mode
    for ch in dataset.channel.values:
        for i in range(8):
            sensor = dataset.sel(channel=ch, col=slice(i*sensor_size,(i+1)*sensor_size))
            background.setdefault(ch, dict())
            background[ch][i] = stats.mode(sensor, axis=None)[0][0]
        avg_background = int(round(np.mean([*background[ch].values()])))
        print('Channel', ch,'::Average background', avg_background)
        # Calculate background correction
        for i in range(8):
            background[ch][i] = avg_background-background[ch][i]

    # Save background correction values in config file
    config = configparser.ConfigParser()
    config.read_dict(background)
    if save_path is None:
        save_path = im_path
    if not machine:
        bg_path = path.join(save_path,'background.cfg')
    else:
        bg_path = path.join(save_path,machine+'.cfg')
    with open(bg_path, 'w') as configfile: config.write(configfile)

def get_HiSeqImages(image_path=None, common_name='', logger = None):


    ims = HiSeqImages(image_path, common_name, logger)
    n_images = len(ims.im)
    if n_images > 1:
        return [HiSeqImages(im=i) for i in ims.im]
    elif n_images == 1:
        ims.im = ims.im[0]
        return ims



class HiSeqImages():
    """HiSeqImages

       **Attributes:**
        - im (dict): List of DataArray from sections
        - channel_color (dict): Dictionary of colors to display each channel as
        - channel_shift (dict): Dictionary of how to register each channel
        - stop (bool): Flag to close viewer
        - app (QTWidget): Application instance

    """

    def __init__(self, image_path=None, common_name='',  im=None,
                       obj_stack=None, RoughScan = False, logger = None):
        """The constructor for HiSeq Image Datasets.

           **Parameters:**
            - image_path (path): Path to images with tiffs

           **Returns:**
            - HiSeqImages object: Object to manipulate and view HiSeq image data

        """

        self.im = []
        self.channel_color = {558:'blue', 610:'green', 687:'magenta', 740:'red'}
        self.channel_shift = {558:[93,None,0,None],
                              610:[90,-3,0,None],
                              687:[0,-93,0,None],
                              740:[0,-93,0,None]}
        self.stop = False
        self.app = None
        self.viewer = None
        self.logger = logger

        if len(common_name) > 0:
            common_name = '*'+common_name

        section_names = []
        if im is None:
            if image_path is None:
                image_path = getcwd()

            # Open zarr
            if image_path[-4:] == 'zarr':
                section_names = self.open_zarr([image_path])

            if obj_stack is not None:
                # Open obj stack (jpegs)
                #filenames = glob.glob(path.join(image_path, common_name+'*.jpeg'))
                n_frames = self.open_objstack(obj_stack)

            elif RoughScan:
                # RoughScans
                filenames = glob.glob(path.join(image_path,'*RoughScan*.tiff'))
                if len(filenames) > 0:
                    n_tiles = self.open_RoughScan(filenames)

            else:
                # Open tiffs
                filenames = glob.glob(path.join(image_path, common_name+'*.tiff'))
                if len(filenames) > 0:
                    section_names = self.open_tiffs(filenames)

                # Open zarrs
                filenames = glob.glob(path.join(image_path, common_name+'*.zarr'))
                if len(filenames) > 0:
                    section_names = self.open_zarr(filenames)

            if len(section_names) > 0:
                message(self.logger, 'Opened', *section_names)

        else:
            self.im = im


    def correct_background(self):

        # Open background config
        config = configparser.ConfigParser()
        if not self.im.machine:
            config_path = pkg_resources.path(recipes, 'background.cfg')
        else:
            config_path = pkg_resources.path(recipes, machine+'.cfg')
        with config_path as config_path_:
            config.read(config_path_)

        # Apply background correction
        ch_list = []
        ncols = len(self.im.col)
        for ch in self.im.channel.values:
            bg_ = np.zeros(ncols)
            i = 0
            for c in range(int(ncols/256)):
                if c == 0:
                    i = self.im.first_group
                if i == 8:
                    i = 0
                bg = config.getint(str(ch),str(i))
                bg_[c*256:(c+1)*256] = bg
                i += 1
            ch_list.append(self.im.sel(channel=ch)+bg_)
        self.im = xr.concat(ch_list,dim='channel')


    def register_channels(self, im):
        """Register image channels."""

        shifted=[]
        for ch in self.channel_shift.keys():
            shift = self.channel_shift[ch]
            shifted.append(im.sel(channel = ch,
                                  row=slice(shift[0],shift[1]),
                                  col=slice(shift[2],shift[3])
                                   ))
        im = xr.concat(shifted, dim = 'channel')
        im = im.sel(row=slice(64,None))                                         # Top 64 rows have white noise

        return im

    def quit(self):
        if self.stop:
            self.app.quit()
            self.viewer.close()
            self.viewer = None

    def hs_napari(self, dataset):

        with napari.gui_qt() as app:
            viewer = napari.Viewer()
            self.viewer = viewer
            self.app = app

            self.update_viewer(dataset)
            start = time.time()

            # timer for exiting napari
            timer = QTimer()
            timer.timeout.connect(self.quit)
            timer.start(1000*1)

            @viewer.bind_key('x')
            def crop(viewer):
                if 'Shapes' in viewer.layers:
                    bound_box = np.array(viewer.layers['Shapes'].data).squeeze()
                else:
                    bound_box = np.array(False)

                if bound_box.shape[0] == 4:

                    #crop full dataset
                    self.crop_section(bound_box)
                    #save current selection
                    selection = {}
                    for d in self.im.dims:
                        if d not in ['row', 'col']:
                            if d in dataset.dims:
                                selection[d] = dataset[d]
                            else:
                                selection[d] = dataset.coords[d].values
                    # update viewer
                    cropped = self.im.sel(selection)
                    self.update_viewer(cropped)


    def show(self, selection = {}, show_progress = True):
        """Display a section from the dataset.

           **Parameters:**
            - selection (dict): Dimension and dimension coordinates to display

        """

        dataset  = self.im.sel(selection)

        if show_progress:
            with ProgressBar() as pbar:
                self.hs_napari(dataset)
        else:
            self.hs_napari(dataset)

    def downscale(self, scale=None):
        if scale is None:
            size_Mb = self.im.size*16/8/(1e6)
            scale = int(2**round(log2(size_Mb)-10))

        if scale > 256:
            scale = 256

        if scale > 0:
            self.im = self.im.coarsen(row=scale, col=scale, boundary='trim').mean()
            self.im.attrs['scale']=scale


    def crop_section(self, bound_box):
        """Return cropped full dataset with intact pixel groups.

           **Parameters:**
            - bound_box (list): Px row min, px row max, px col min, px col max

           **Returns:**
            - dataset: Row and column cropped full dataset
            - int: Initial pixel group index of dataset

        """

        if bound_box.shape[1] >= 2:
            bound_box = bound_box[:,-2:]

        nrows = len(self.im.row)
        ncols = len(self.im.col)
        #pixel group scale
        pgs = int(256/self.im.attrs['scale'])
        #tile scale
        ts = int(pgs*8)

        row_min = int(round(bound_box[0,0]))
        if row_min < 0:
            row_min = 0

        row_max = int(round(bound_box[1,0]))
        if row_max > nrows:
            row_max = nrows

        col_min = bound_box[0,1]
        if col_min < 0:
            col_min = 0
        col_min = int(floor(col_min/pgs)*pgs)

        col_max = bound_box[2,1]
        if col_max > ncols:
            col_max = ncols
        col_max = int(ceil(col_max/pgs)*pgs)

        group_index = floor((col_min + self.im.first_group*pgs)%ts/ts*8)

        self.im = self.im.sel(row=slice(row_min, row_max),
                              col=slice(col_min, col_max))
        self.im.attrs['first_group'] = group_index

    def update_viewer(self, dataset):

        viewer = self.viewer
        # Delete old layers
        for i in range(len(viewer.layers)):
            viewer.layers.pop(0)

        # Display only 1 layer if there is only 1 channel
        channels = dataset.channel.values

        if not channels.shape:
            ch = int(channels)
            message(self.logger, 'Adding', ch, 'channel')
            layer = viewer.add_image(dataset.values,
                                     colormap=self.channel_color[ch],
                                     name = str(ch),
                                     blending = 'additive')
        else:
            for ch in channels:
                message(self.logger, 'Adding', ch, 'channel')
                layer = viewer.add_image(dataset.sel(channel = ch).values,
                                         colormap=self.channel_color[ch],
                                         name = str(ch),
                                         blending = 'additive')


    def save_zarr(self, save_path):
        """Save all sections in a zipped zarr store.

           Note that coordinates for unused dimensions are not saved.

           **Parameters:**
            - save_path (path): directory to save store

        """

        if not path.isdir(save_path):
            mkdir(save_path)

        save_name = path.join(save_path,self.im.name+'.zarr')
        # Remove coordinate for unused dimensions
        for c in self.im.coords.keys():
            if c not in self.im.dims:
                self.im = self.im.reset_coords(names=c, drop=True)

        self.im.to_dataset().to_zarr(save_name)

        # save attributes
        f = open(self.im.name+'.attrs',"w")
        for key, val in self.im.attrs.items():
            f.write(str(key)+' '+str(val)+'\n')
        f.close()

    def open_zarr(self, filenames):
        """Create labeled dataset from zarrs.

           **Parameters:**
           - filename(list): List of full file path names to images

           **Returns:**
           - array: Labeled dataset

        """

        im_names = []
        for fn in filenames:
            im_name = path.basename(fn)[:-5]
            im = xr.open_zarr(fn).to_array()
            im = im.squeeze().drop_vars('variable').rename(im_name)
            im_names.append(im_name)

            # add attributes
            attr_path = fn[:-5]+'.attrs'

            if path.exists(attr_path):
                attrs = {}
                with open(attr_path) as f:
                    for line in f:
                        items = line.split()
                        if len(items) == 2:
                            try:
                                value = int(items[1])
                            except:
                                value = items[1]
                        else:
                            value = ''
                        im.attrs[items[0]] = value

            self.im.append(im)

        return im_names


    def open_RoughScan(self,filenames):
        # Open RoughScan tiffs
        comp_sets = dict()
        for fn in filenames:
            # Break up filename into components
            comp_ = path.basename(fn)[:-5].split("_")
            for i, comp in enumerate(comp_):
                comp_sets.setdefault(i,set())
                comp_sets[i].add(comp)

        shape = imageio.imread(filenames[0]).shape
        lazy_arrays = [dask.delayed(imageio.imread)(fn) for fn in filenames]
        lazy_arrays = [da.from_delayed(x, shape=shape, dtype='int16') for x in lazy_arrays]
        #images = [imageio.imread(fn) for fn in filenames]

        # Organize images
        #0 channel, 1 RoughScan, 2 x_step, 3 obj_step
        fn_comp_sets = list(comp_sets.values())
        for i in [0,2]:
            fn_comp_sets[i] = [int(x[1:]) for x in fn_comp_sets[i]]
        fn_comp_sets = list(map(sorted, fn_comp_sets))
        remap_comps = [fn_comp_sets[0], [1], fn_comp_sets[2]]
        a = np.empty(tuple(map(len, remap_comps)), dtype=object)
        for fn, x in zip(filenames, lazy_arrays):
            comp_ = path.basename(fn)[:-5].split("_")
            channel = fn_comp_sets[0].index(int(comp_[0][1:]))
            x_step = fn_comp_sets[2].index(int(comp_[2][1:]))
            a[channel, 0, x_step] = x


        # Label array
        dim_names = ['channel', 'row', 'col']
        channels = [int(ch) for ch in fn_comp_sets[0]]
        coord_values = {'channel':channels}
        im = xr.DataArray(da.block(a.tolist()),
                               dims = dim_names,
                               coords = coord_values,
                               name = 'RoughScan')

        im = im.assign_attrs(first_group = 0, machine='', scale=1)
        self.im = im.sel(row=slice(64,None))

        return len(fn_comp_sets[2])

    def open_objstack(self, obj_stack):


        # Open jpegs
        # comp_sets = dict()
        # for fn in filenames:
        #     # Break up filename into components
        #     comp_ = path.basename(fn)[:-5].split("_")
        #     for i, comp in enumerate(comp_):
        #         comp_sets.setdefault(i,set())
        #         comp_sets[i].add(comp)
        #
        #
        # lazy_arrays = [dask.delayed(imageio.imread)(fn) for fn in filenames]
        # lazy_arrays = [da.from_delayed(x, shape=(16,2048), dtype='int8') for x in lazy_arrays]
        #
        # # Organize images
        # #0 channel, 1 frame
        # fn_comp_sets = list(comp_sets.values())
        # for i in [0,1]:
        #     fn_comp_sets[i] = [int(x) for x in fn_comp_sets[i]]
        #     fn_comp_sets[i] = sorted(fn_comp_sets[i])
        # remap_comps = [fn_comp_sets[0], fn_comp_sets[1], [1], [1]]
        # a = np.empty(tuple(map(len, remap_comps)), dtype=object)
        # for fn, x in zip(filenames, lazy_arrays):
        #     comp_ = path.basename(fn)[:-5].split("_")
        #     channel = fn_comp_sets[0].index(int(comp_[0]))
        #     frame = fn_comp_sets[1].index(int(comp_[1]))
        #     a[channel, frame, 0, 0] = x


        # Label array
        # dim_names = ['channel', 'frame', 'row', 'col']
        dim_names = ['frame', 'channel', 'row', 'col']
        channels = [687, 558, 610, 740]
        frames = range(obj_stack.shape[0])
        coord_values = {'channel':channels, 'frame':frames}
        im = xr.DataArray(obj_stack.tolist(),
                               dims = dim_names,
                               coords = coord_values,
                               name = 'Objective Stack')

        im = im.assign_attrs(first_group = 0, machine = '', scale=1)
        self.im = im

        return obj_stack.shape[0]


    def open_tiffs(self, filenames):
        """Create labeled dataset from tiffs.

           **Parameters:**
           - filename(list): List of full file path names to images

           **Returns:**
           - array: Labeled dataset

        """

        # Open tiffs
        section_sets = dict()
        section_meta = dict()
        for fn in filenames:
            # Break up filename into components
            comp_ = path.basename(fn)[:-5].split("_")
            if len(comp_) == 6:
                section = comp_[2]
                # Add new section
                if section_sets.setdefault(section, dict()) == {}:
                    im = imageio.imread(fn)
                    section_meta[section] = {'shape':im.shape,'dtype':im.dtype,'filenames':[]}

                for i, comp in enumerate(comp_):
                    # Add components
                    section_sets[section].setdefault(i, set())
                    section_sets[section][i].add(comp)
                    section_meta[section]['filenames'].append(fn)

        im_names = []
        for s in section_sets.keys():
            # Lazy open images
            filenames = section_meta[s]['filenames']
            lazy_arrays = [dask.delayed(imageio.imread)(fn) for fn in filenames]
            shape = section_meta[s]['shape']
            dtype = section_meta[s]['dtype']
            lazy_arrays = [da.from_delayed(x, shape=shape, dtype=dtype) for x in lazy_arrays]

            # Organize images
            #0 channel, 1 flowcell, 2 section, 3 cycle, 4 x position, 5 objective position
            fn_comp_sets = list(section_sets[s].values())
            for i in [0,3,4,5]:
                fn_comp_sets[i] = [int(x[1:]) for x in fn_comp_sets[i]]
                fn_comp_sets[i] = sorted(fn_comp_sets[i])
            remap_comps = [fn_comp_sets[0], fn_comp_sets[3], fn_comp_sets[5], [1],  fn_comp_sets[4]]
            a = np.empty(tuple(map(len, remap_comps)), dtype=object)
            for fn, x in zip(filenames, lazy_arrays):
                comp_ = path.basename(fn)[:-5].split("_")
                channel = fn_comp_sets[0].index(int(comp_[0][1:]))
                cycle = fn_comp_sets[3].index(int(comp_[3][1:]))
                obj_step = fn_comp_sets[5].index(int(comp_[5][1:]))
                x_step = fn_comp_sets[4].index(int(comp_[4][1:]))
                a[channel, cycle, obj_step, 0, x_step] = x

            # Label array
            dim_names = ['channel', 'cycle', 'obj_step', 'row', 'col']
            coord_values = {'channel':fn_comp_sets[0], 'cycle':fn_comp_sets[3], 'obj_step':fn_comp_sets[5]}
            im = xr.DataArray(da.block(a.tolist()),
                                   dims = dim_names,
                                   coords = coord_values,
                                   name = s[1:])


            im = self.register_channels(im.squeeze())
            im = im.assign_attrs(first_group = 0, machine = '', scale=1)
            self.im.append(im)
            im_names.append(s[1:])

        return im_names

class HiSeqImagesOLD():
    """HiSeqImages

       **Attributes:**
        - sections (dict): Dictionary of sections
        - channel_color (dict): Dictionary colors to display each channel as

    """

    def __init__(self, image_path):
        """The constructor for HiSeq Image Datasets.

           **Parameters:**
            - image_path (path): Path to images with tiffs

           **Returns:**
            - HiSeqImages object: Object to manipulate and view HiSeq image data

        """
        self.sections = {}
        self.channel_color = {558:'blue', 610:'green', 687:'magenta', 740:'red'}

        # Get number of tiffs and zarrs
        all_names = listdir(image_path)
        n_tiffs = len([name for name in all_names if '.tiff' in name])
        n_zarrs = len([name for name in all_names if '.zarr' in name])

        # Convert tiffs to zarr
        if n_tiffs > 0 & n_zarrs == 0:
            zarr_path = tiff2zarr(image_path)
            n_zarrs = 1
            image_path = zarr_path
        elif n_tiffs > 0 & n_zarrs > 0:
            print('Detected both zarr and tiff images, using only zarr')

        dataset = None
        if n_zarrs > 0:
            df = get_image_df(image_path, im_type = '.zarr')
            if len(df) > 0:
                sections = [*set(df.s)]
                for s in sections:
                    self.sections[s[1:]] = label_images(df[df.s == s], image_path)
