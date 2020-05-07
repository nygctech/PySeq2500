#!/usr/bin/python

import numpy as np
import pandas as pd
from math import log2
from os import listdir, stat, path
from scipy import stats
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
from skimage import io
from skimge.exposure import match_histograms
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, square



def get_image_df(dir, image_name = None):
  '''Get dataframe of images.

     Dataframe columns: channel, flowcell, section, cycle, x, o.
     * channel: Emission channel of the image.
     * flowcell: A or B.
     * section: Name of imaged area.
     * cycle: Nth imaging round of the section.
     * x: X-stage position of the image.
     * o: Objective stage position of the image.

     Parameters:
     dir (path): Directory where images are stored.
     image_name (str): Name common to all images.

     Return
     dataframe: Image meta data with image names as index.

  '''

    all_names = os.listdir(dir)
    if image_name is None:
      image_names = all_names
    else:
      image_names = [name for name in all_names if image_name in name]

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



def norm_and_stitch(dir, df_x, overlap = 0, scaled = False):
  '''Normalize and stitch scans.

     Images are normalized by matching histogram to first strip in first image.
     The scale factor of the stitched image is saved in the first pixel
     (row = 0, column = 0).


     Parameters
     dir (path): Directory where image scans are stored.
     df_x (df): Dataframe of metadata of image scans to stitch.
     scaled (bool): If True autoscale images to ~2^18 bytes, if False do not scale.

     Return:
     array: Normalized, stitched, and downscaled image as numpy array.

     TODO:
     * stich images with an overlap

  '''

  df_x.sort_values(by=['x'])
  scans = []                                                                    # list of xpos scans
  ref = None
  scale_factor = None
  for name in df_x.index:
    im = io.imread(path.join(dir,name))
    im = im[64:]                                                                # Remove whiteband artifact

    if scaled = True:
        # Scale images so they are ~ 256 kb
        if scale_factor is None :
            size = stat(path.join(dir,name)).st_size
            scale_factor = (2**(log2(size)-18))**0.5
            scale_factor = round(log2(scale_factor))

        im = downscale_local_mean(im, (scale_factor, scale_factor))
    else:
        scale_factor = 1

    x_px = int(im.shape[1]/8)

    for i in range(8):
      # Stitch images
      sub_im = im[:,(i)*x_px:(i+1)*x_px]

      if ref is None:
        # Make first strip reference for histogram matching
        ref = sub_im
        plane = sub_im
      else:
        sub_im = match_histograms(sub_im, ref)
        plane = np.append(plane, sub_im, axis = 1)

  plane = plane.astype('uint8')
  plane = img_as_ubyte(plane)
  plane[0,0] = scale_factor

  return plane


 def avg_images(images):
     '''Average pixel values in images

        First image is used as a reference.
        All other image histograms are matched to the reference.
        Then the images are averaged together.

        Parameters
        images (list): List of images to sum.

        Return:
        array: Numpy array of average image.

     '''

    sum_im = None
    ref = None
    for im in images:
        if ref is None:
            ref = im

        if sum_im is None:
            sum_im = im
        else:
            im = match_histograms(im, ref)
            sum_im = np.add(sum_im,im)

    sum_im = sum_im/(len(im))
    sum_im = sum_im.astype('uint8')

def get_roi(im, psat = 98, sq_size = 3):
    '''Get region of interest.

       Parameters:
       im (array): Image as a numpy array.
       psat (int): Cut off percentile for saturated pixels.
       sq_size (int): Size of square to remove small artifacts in segmented
       image.

       Returns
       array: Binary image as numpy array with region of interest as 1.

    '''

    # Make background 0, assume background is most frequent px value
    p_back = stats.mode(im, axis=None)
    im[im <= p_back[0]] = 0

    # Make saturated pixels 0
    p_sat = np.percentile(im, (psat,))
    im[im > p_sat] = 0

    # Make elevation map
    elev_map = sobel(im)

    # Get markers
    markers = np.zeros_like(im)
    p_sign = np.percentile(im, (98,))
    markers[im == 0] = 1
    markers[im > p_sign] = 2

    # Get segmented image
    im = morphology.watershed(elev_map, markers)

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

    return roi


def get_focus_pos(roi):
    '''Get focus positions.

       Parameters:
       roi (array): Binary array of region of intereste

       Returns
       array: Coordinates of 4 focus pixels, 1st 3 points are edge positions,
       the last point is the centroid.

    '''

    # Get properties of region of interest
    roi_props = regionprops(roi)
    center = props[0].centroid

    # Get edge of region of interest
    contour = find_contours(roi, 0.5)
    contour = np.array(contour[0], dtype = 'int')
    contour = np.unique(contour, axis = 0)

    # Find 2 points along edge the maximize distance
    dist = cdist(contour, contour)
    max_ind = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    max_ind = [*max_ind]

    # Find 3rd point along edge that maximizes distance
    dist2 = np.append(dist[max_ind[0],:],dist[max_ind[1],:])
    dist2 = np.reshape(dist2, (2,contour.shape[0]))
    dist2 = np.sum(dist2, axis = 0)
    max_ind.append(np.argmax(dist2))

    # Reorder stage points to match z stage motor index
    focus_pos = np.zeros(shape=[3,2])
    # Motor position 0
    m0 = np.where(pos[:,0] == np.min(pos[:,0]))[0][0]
    focus_pos[0,:] = pos[m0,:]
    pos = np.delete(pos,m0,0)
    # Motor position 1
    m1 = np.where(pos[:,1] == np.min(pos[:,1]))[0][0]
    focus_pos[1,:] = pos[m1,:]
    pos = np.delete(pos,m1,0)
    # Motor position 2
    focus_pos[2,:] = pos[0,:]
    # Center position
    focus_pos[3,:] = props[0].centroid

    return focus_pos

def shift_focus(fpoint, center, scale):
    '''Shift focus points towards the center.

       Parameters:
       fpoint (int,int): Pixel row and column coordinates of a focal point.
       center (int, int): Pixel row and column coordinates of the center point.
       scale (float): Scale to shift a focal point towards the center.

       Returns
       [int,int]: Pixel row and column coordinates of focal point shifted
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
