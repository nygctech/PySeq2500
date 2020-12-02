#!/usr/bin/python

import numpy as np
import pandas as pd
from math import log2
from os import listdir, stat, path, getcwd
from scipy import stats
from scipy.spatial.distance import cdist
from skimage.exposure import match_histograms
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
import imageio

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


def get_image_df(image_path = None, image_name = None):
    """Get dataframe of rough focus images.

       **Parameters:**
       image_path (path): Directory where images are stored.
       image_name (str): Name common to all images.

       **Returns:**
       dataframe: Dataframe of image metadata with image names as index.

    """

    if image_path is None:
        image_path = getcwd()

    all_names = listdir(image_path)
    if image_name is None:
      image_names = [name for name in all_names if '.tiff' in name]
    else:
      im_names = [name for name in all_names if image_name in name]
      image_names = [name for name in im_names if '.tiff' in name]

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

def sum_images(images, thresh = 81, logger = None):
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
    # ref = None

    try:
        thresh = float(thresh)
    except:
        thresh = 81.0

    # Select image with largest signal to noise as reference
    #SNR = np.array([])
    # i = 0
    for c, im in enumerate(images):
        #kurt_z, pvalue = stats.kurtosistest(im, axis = None)
        #kurt_z = stats.kurtosis(im, axis=None)
        k = kurt(im)
        message(logger, name_, 'Channel',c, 'k = ', k)
        if k > thresh:
            #message(logger, name_, 'Signal in channel',i)
            # Add add image
            if sum_im is None:
                sum_im = im.astype('int16')
            else:
                sum_im = np.add(sum_im, im)
            #SNR = np.append(SNR,np.mean(im)/np.std(im))
        # else:
        #     message(logger, name_, 'No signal in channel',i)
        #     # Remove images without signal
        #     #SNR = np.append(SNR, 0)

    #     i += 1
    #
    # ims = []
    # for i, s in enumerate(SNR):
    #   if s > 0:
    #     ims.append(images[i])
    # SNR = SNR[SNR > 0]
    #
    # if SNR.size > 0:
    #     ref_i = np.argmax(SNR)
    #     ref = images[ref_i]
    #
    #     # Sum images
    #     for i, im in enumerate(ims):
    #         # Match histogram to reference image
    #         if i != ref_i:
    #             _im = im
    #             #_im = match_histograms(im, ref)
    #         else:
    #             _im = im
    #
    #         # Add add image
    #         if sum_im is None:
    #             sum_im = _im.astype('uint16')
    #         else:
    #             sum_im = np.add(sum_im, _im)

    return sum_im

def kurt(im):
    """Return kurtosis = mean((image-mode)/2)^4). """

    im = im.astype('int16')
    mode = stats.mode(im, axis = None)[0][0]
    z_score = (im-mode)/2
    k = np.mean(z_score**4)

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
          dist2 = np.append(dist[prev2,:],dist[prev1,:], axis =0)
          ind = np.argmax(np.sum(dist2))
          ord_points[i,:] = _markers[ind,:]
          dist = np.delete(dist,ind,1)
          _markers = np.delete(_markers,ind, axis=0)
    else:
        ord_points = c_markers

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
