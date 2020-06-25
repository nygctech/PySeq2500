#!/usr/bin/python

import numpy as np
import pandas as pd
from math import log2
from os import listdir, stat, path
from scipy import stats
from scipy.spatial.distance import cdist
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
from skimage import io
from skimage.exposure import match_histograms
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, square, watershed
from skimage.filters import sobel


def get_image_df(im_path, image_name = None):
  '''Get dataframe of images.

     Dataframe columns: channel, flowcell, section, cycle, x, o.
     * channel: Emission channel of the image.
     * flowcell: A or B.
     * section: Name of imaged area.
     * cycle: Nth imaging round of the section.
     * x: X-stage position of the image.
     * o: Objective stage position of the image.

     Parameters:
     - dir (path): Directory where images are stored.
     - image_name (str): Name common to all images.

     Returns:
     - dataframe: Image meta data with image names as index.

  '''

  all_names = listdir(im_path)
  if image_name is None:
    im_names = all_names
    image_names = [name for name in im_names if '.tiff' in name]
  else:
    im_names = [name for name in all_names if image_name in name]
    image_names = [name for name in im_names if '.tiff' in name]

  # Dataframe for metdata
  metadata = pd.DataFrame(columns = ('channel','flowcell','section',
                                     'cycle','x','o'))

  # Extract metadata
  for name in image_names:

    meta = name[:-5].split('_')

    # Convert channels to int
    meta[0] = int(meta[0])
    # Remove c from cycle
    meta[3] = int(meta[4][1:])
    # Remove x from xposition
    meta[4] = int(meta[5][1:])
    # Remove o from objective position
    meta[5] = int(meta[6][1:])

    metadata.loc[name] = meta


  metadata.sort_values(by=['flowcell', 'section', 'cycle', 'channel',
                           'o','x'])

  return metadata



def norm_and_stitch(im_path, df_x, overlap = 0, scaled = False):
  '''Normalize and stitch scans.

     Images are normalized by matching histogram to first strip in first image.
     The scale factor of the stitched image is saved in the first pixel
     (row = 0, column = 0

     Parameters:
     - dir (path): Directory where image scans are stored.
     - df_x (df): Dataframe of metadata of image scans to stitch.
     - scaled (bool): If True autoscale images to ~2^18 bytes, if False do not scale.

     Return:
     - array: Normalized, stitched, and downscaled image as numpy array.

     TODO:
     * stich images with an overlap

  '''

  df_x.sort_values(by=['x'])
  scans = []                                                                    # list of xpos scans
  ref = None
  scale_factor = None
  for name in df_x.index:
    im = io.imread(path.join(im_path,name))
    im = im[64:]                                                                # Remove whiteband artifact

    if scaled:
        # Scale images so they are ~ 256 kb
        if scale_factor is None :
            size = stat(path.join(im_path,name)).st_size
            scale_factor = (2**(log2(size)-18))**0.5
            scale_factor = round(log2(scale_factor))
            print(scale_factor)

        im = downscale_local_mean(im, (scale_factor, scale_factor))
    else:
        scale_factor = 1

    x_px = int(im.shape[1]/8)

    for i in range(8):
      # Stitch images
      sub_im = im[:,(i)*x_px:(i+1)*x_px]

      if ref is None:
        # Make first strip reference for histogram matching
        ref = im[:,7*x_px:8*x_px]
        plane = match_histograms(sub_im, ref)
      else:
        sub_im = match_histograms(sub_im, ref)
        plane = np.append(plane, sub_im, axis = 1)

  plane = plane.astype('uint8')
  plane = img_as_ubyte(plane)
  plane[0,0] = scale_factor

  return plane
from skimage.util import img_as_ubyte

def sum_images(images):
    '''Average pixel values in images.

       Image with largest signal to noise used a reference.
       Images without significant kurtosis are discarded.
       Remaining image histograms are matched to the reference.
       Then the images are summed together.

       Parameters:
       - images (list): List of images to sum.

       Return:
       - array: Numpy array of average image.

    '''

    sum_im = None
    ref = None

    # Select image with largest signal to noise as reference
    SNR = np.array([])
    i = 0
    for im in images:
        kurt_z, pvalue = stats.kurtosistest(im, axis = None)
        if kurt_z > 1.96:
            SNR = np.append(SNR,np.mean(im)/np.std(im))
        else:
            # Remove images without signal
            images.pop(i)
        i += 1
    ref_i = np.argmax(SNR)
    ref = images[ref_i]

    # Sum images
    i = 0
    for im in images:
        # Match histogram to reference image
        if i != ref_i:
            _im = match_histograms(im, ref)
        i += 1
        # Add add image
        if sum_im is None:
            sum_im = _im
        else:
            sum_im = np.add(sum_im, _im)

    return sum_im

def get_focus_points(im, scale, min_n_markers, p_sat = 99.5):
    '''Get potential points to focus on.

       The top 1000 brightest pixels that are not saturate are found.
       Then the focus field of views with the top min_n_markers contrast are
       ordered based on the distance from each other, with the points farthest
       away from each other being first.

       Parameters:
       - im (array): Summed image across all channels with signal
       - scale (int): Factor at which the image is scaled down.
       - min_n_markers (int): Minimum number of focus points desired.
       - p_sat (float): Percentile to call pixels saturated

       Returns:
       - array: Row, Column list of ordered pixels to use as focus points

    '''

    px_rows, px_cols = im.shape
    px_sat = np.percentile(sum_im, (p_sat))
    px_score = np.reshape(stats.zscore(sum_im, axis = None), (px_rows, px_cols))

    edge_width = int(2048/scale/2)


    im_ = np.zeros_like(sum_im)
    px_score_thresh = 3
    while np.sum(im_ != 0) == 0:
        # Get brightest pixels
        im_[px_score > px_score_thresh] = sum_im[px_score > px_score_thresh]
        # Remove "saturated" pixels
        im_[im > px_sat] = 0
        #Remove Edges
        if edge_width < px_cols/2:
          im_[0:edge_width,:] = 0
          im_[:, px_cols-edge_width:px_cols] = 0
          im_[px_rows-edge_width:px_rows, :] = 0
          im_[:,0:edge_width] = 0

         px_score_thresh -= 0.5

    markers = np.argwhere(im_ != 0)


    # Subset to 1000 points
    if len(markers) > 1000:
      rand_markers = np.random.choice(range(n_markers), size = 1000)
      markers = markers[rand_markers,:]
    n_markers = len(markers)

    # Compute contrast
    c_score = np.zeros_like(markers[:,1])
    for row in range(n_markers):
      mark = markers[row,:]
      if edge_width < px_cols/2:
        frame = im[mark[0],mark[1]-edge_width:mark[1]+edge_width]
      else:
        frame = im[mark[0],:]
      c_score[row] = np.max(frame) - np.min(frame)


    #resolution = 0.7
    #marker_thresh = 3 + int((px_rows*px_cols*scale**2)**0.5*resolution/1000)*10
    # Get theminimum number of markers needed with the highest contrast
    p_top = (1 - min_n_markers/n_markers)*100
    c_cutoff = np.percentile(c_score, p_top)
    c_markers = markers[c_score >= c_cutoff,:]

    # Compute distance matrix
    dist = cdist(c_markers, c_markers)
    max_ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)          #returns tuple

    # Order marker points based on distance from each other
    # Markers farthest from each other are first
    n_markers = len(c_markers)
    ord_points = np.zeros_like(c_markers)
    ord_points[0,:] = c_markers[max_ind[0],:]
    ord_points[1,:] = c_markers[max_ind[1],:]
    _markers = np.copy(c_markers)
    prev2 = max_ind[0]
    prev1 = max_ind[1]
    dist = np.delete(dist,[prev2,prev1],1)
    _markers = np.delete(_markers,[prev2,prev1], axis=0)
    for i in range(2,n_markers-2):
      dist2 = np.append(dist[prev2,:],dist[prev1,:], axis =0)
      ind = np.argmax(np.sum(dist2))
      ord_points[i,:] = _markers[ind,:]
      dist = np.delete(dist,ind,1)
      _markers = np.delete(_markers,ind, axis=0)

    return _markers


def get_roi(im, per_sat = 98, sq_size = 3):
    '''Get region of interest.

       Parameters:
       - im (array): Image as a numpy array.
       - psat (int): Cut off percentile for saturated pixels.
       - sq_size (int): Size of square to remove small artifacts in segmented
                        image.

       Returns:
       - array: Binary image as numpy array with region of interest as 1.

    '''

    # Make elevation map
    elev_map = sobel(im)

    # Get markers
    markers = np.zeros_like(im)
    p_sign = np.percentile(im, (per_sat,))
    markers[im == 0] = 1
    markers[im > p_sign] = 2

    # Get segmented image
    im = watershed(elev_map, markers)

    # remove border objects
    im = clear_border(im)
    # remove small objects
    im = closing(im, square(sq_size))

    # label regions
    labeled = label(im)
    # choose largest region that is not backround
    roi_id = stats.mode(labeled[labeled != 0], axis=None)
    # Create region of interest binary image
    roi = np.zeros_like(im)
    roi[labeled == roi_id[0]] = 1
    roi = roi.astype('uint8')

    return roi


def get_focus_pos(roi):
    '''Get focus positions.

       Parameters:
       - roi (array): Binary array of region of intereste

       Returns
       - array: Coordinates of 4 focus pixels, 1st 3 points are edge positions,
                the last point is the centroid.

    '''

    # Get properties of region of interest
    roi_props = regionprops(roi)
    center = roi_props[0].centroid

    # Get edge of region of interest
    contour = find_contours(roi, 0.5)
    contour = np.array(contour[0], dtype = 'int')
    contour = np.unique(contour, axis = 0)
    n_points = contour.shape[0]
    if n_points > 1000:
        sample_ind = np.random.choice(np.arange(n_points), size = 1000)
        contour = contour[sample_ind,:]

    # Find 2 points along edge the maximize distance
    dist = cdist(contour, contour)
    max_ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)          #returns tuple
    max_ind = [*max_ind]                                                        #converts into list

    # Find 3rd point along edge that maximizes distance
    dist2 = np.append(dist[max_ind[0],:],dist[max_ind[1],:])
    dist2 = np.reshape(dist2, (2,contour.shape[0]))
    dist2 = np.sum(dist2, axis = 0)
    max_ind.append(np.argmax(dist2))

##    # Reorder stage points to match z stage motor index
##    focus_pos = np.zeros(shape=[3,2])
##    # Motor position 0
##    m0 = np.where(pos[:,0] == np.min(pos[:,0]))[0][0]
##    focus_pos[0,:] = pos[m0,:]
##    pos = np.delete(pos,m0,0)
##    # Motor position 1
##    m1 = np.where(pos[:,1] == np.min(pos[:,1]))[0][0]
##    focus_pos[1,:] = pos[m1,:]
##    pos = np.delete(pos,m1,0)
##    # Motor position 2
##    focus_pos[2,:] = pos[0,:]
##    # Center position
    focus_pos = contour[max_ind,:]
    center = np.array(center, dtype ='int')
    center = np.reshape(center, (1,2))
    focus_pos = np.append(focus_pos, center, axis =0)

    return focus_pos


def shift_focus(fpoint, center, scale):
    '''Shift focus points towards the center.

       Parameters:
       - fpoint (int,int): Pixel row and column coordinates of a focal point.
       - center (int, int): Pixel row and column coordinates of the center point.
       - scale (float): Scale to shift a focal point towards the center.

       Returns
       - [int,int]: Pixel row and column coordinates of focal point shifted
                    towards center.

    '''

    # c is column
    # r is row
    c0 = center[1]
    r0 = center[0]
    c1 = fpoint[1]
    r1 = fpoint[0]

    # Find unit vector from focal point to center
    u_c = c0-c1
    u_r = r0-r1
    mag = (u_r**2 + u_c**2)**0.5
    u_c = u_c / mag
    u_r = u_r / mag

    # orientation of focal point with respect to center
    orientation = abs(u_r)

    # column and row of shifted focal point
    c = int(c1 + u_c*scale*orientation)
    r = int(r1 + u_r*scale*orientation)

    return [r, c]
