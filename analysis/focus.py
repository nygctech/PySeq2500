#!/usr/bin/python
import pandas as pd
import numpy as np
from numpy.linalg import svd
import image
from scipy.optimize import least_squares
import os
from skimage import io
from os import path, listdir, stat
from scipy import stats
from math import log2
from skimage.exposure import match_histograms
from skimage.util import img_as_ubyte
from skimage.transform import downscale_local_mean
import imageio
import importlib
importlib.reload(image)

def calibrate(hs, pos_list):
    for pos_i, pos_dict in enumerate(pos_list):
        # Initial scan of section
        hs.y.move(pos_dict['y_initial'])
        hs.x.move(pos_dict['x_initial'])
        hs.z.move([21500, 21500, 21500])
        obj_pos = int((hs.obj.focus_stop - hs.obj.focus_start)/2 + hs.obj.focus_start)
        hs.obj.move(obj_pos)
        rough_ims, scale = rough_focus(hs, pos_dict['n_tiles'], pos_dict['n_frames'], 'RoughScan'+str(pos_i))
        for i in range(len(hs.channels)):
            imageio.imwrite(path.join(hs.image_path,'c'+str(hs.channels[i])+'INIT'+str(pos_i)+'.tiff'), rough_ims[i])

        # Sum channels with signal
        sum_im = image.sum_images(rough_ims)
        imageio.imwrite(path.join(hs.image_path,'sum_im' + str(pos_i) + '.tiff'), sum_im)
    ##    sum_im = imageio.imread(path.join(hs.image_path,'sum_im.tiff'))
    ##    scale = 16

        # Find pixels to focus on
        px_rows, px_cols = sum_im.shape
        n_markers = 3 + int((px_rows*px_cols*scale**2)**0.5*hs.resolution/1000)
        ord_points = image.get_focus_points(sum_im, scale, n_markers*10)
        np.savetxt(path.join(hs.image_path, 'ord_points'+str(pos_i)+'.txt'), ord_points)

    ##    ord_points = np.loadtxt(path.join(hs.image_path, 'ord_points.txt'))

        #Get stage positions on in-focus points
        focus_points = get_focus_data(hs, ord_points, n_markers, scale, pos_dict)
        np.savetxt(path.join(hs.image_path, 'focus_points'+str(pos_i)+'.txt'), focus_points)
        last_point = focus_points[-1,3]
        np.savetxt(path.join(hs.image_path, 'last_point'+str(pos_i)+'.txt'), np.array([last_point]))
    ##    focus_points = np.loadtxt(path.join(hs.image_path, 'focus_points.txt'))
        pos_ord_points = ord_points[focus_points[:,3].astype(int)]
        np.savetxt(path.join(hs.image_path, 'pos_ord_points'+str(pos_i)+'.txt'), pos_ord_points)

    z_list = [19000, 20000, 23000, 24000]
    for motor in range(3):
        for z in z_list:
            print('moving motor ' + str(motor) + ' to ' + str(z))
            z_pos = [21500, 21500, 21500]
            z_pos[motor] = z
            hs.z.move(z_pos)
            for pos_i, pos_dict in enumerate(pos_list):
                pos_ord_points = np.loadtxt(path.join(hs.image_path,  'pos_ord_points'+str(pos_i)+'.txt'))
                fp_name = ''
                for z_ in hs.z.position:
                    fp_name += str(z_) + '_'
                fp_name += 'm'+str(motor)+'_'+str(pos_i) + '.txt'
                fp = get_focus_data(hs, pos_ord_points, n_markers, scale, pos_dict)
                if focus_points.any():
                    np.savetxt(path.join(hs.image_path, fp_name), fp)
                else:
                    print('no data for motor ' + str(motor) + ' at ' + str(z))

def autofocus(hs, pos_dict):
    # Initial scan of section
    hs.y.move(pos_dict['y_initial'])
    hs.x.move(pos_dict['x_initial'])
    hs.z.move([21500, 21500, 21500])
    obj_pos = int((hs.obj.focus_stop - hs.obj.focus_start)/2 + hs.obj.focus_start)
    hs.obj.move(obj_pos)
    rough_ims, scale = rough_focus(hs, pos_dict['n_tiles'], pos_dict['n_frames'], image_name = 'INIT')
    for i in range(len(hs.channels)):
        imageio.imwrite(path.join(hs.image_path,'c'+str(hs.channels[i])+'INIT.tiff'), rough_ims[i])
####    #rough_ims = []
####    #for i in range(len(hs.channels)):
####    #    rough_ims.append(imageio.imread(path.join(hs.image_path,'c'+str(hs.channels[i])+'RoughFocus.tiff')))
####    #    scale = 16

    # Sum channels with signal
    sum_im = image.sum_images(rough_ims)
    imageio.imwrite(path.join(hs.image_path,'INIT_sum_im.tiff'), sum_im)
####    sum_im = imageio.imread(path.join(hs.image_path,'sum_tilt_im.tiff'))
####    scale = 16

    # Find pixels to focus on
    px_rows, px_cols = sum_im.shape
    n_markers = 3 + int((px_rows*px_cols*scale**2)**0.5*hs.resolution/1000)
    ord_points = image.get_focus_points(sum_im, scale, n_markers*10)
    np.savetxt(path.join(hs.image_path, 'INIT_ord_points.txt'), ord_points)
    #ord_points = np.loadtxt(path.join(hs.image_path, 'INIT_ord_points.txt'))


    # Get stage positions on in-focus points
    focus_points = get_focus_data(hs, ord_points, n_markers, scale, pos_dict)
    np.savetxt(path.join(hs.image_path, 'INIT_focus_points.txt'), focus_points)
    last_point = focus_points[-1,3]
    np.savetxt(path.join(hs.image_path, 'last_point.txt'), np.array([last_point]))

    # Save focus positions
    pos_points = ord_points[focus_points[:,3].astype(int)]

    # Drastic Tilt
    hs.y.move(pos_dict['y_initial'])
    hs.x.move(pos_dict['x_initial'])
    hs.z.move([21000, 22000, 21500])
    rough_ims, scale = rough_focus(hs, pos_dict['n_tiles'], pos_dict['n_frames'], image_name = 'TILT')
    for i in range(len(hs.channels)):
        imageio.imwrite(path.join(hs.image_path,'c'+str(hs.channels[i])+'TILT.tiff'), rough_ims[i])

    # Get stage positions on in-focus points
    focus_points = get_focus_data(hs, pos_points, n_markers, scale, pos_dict)
    np.savetxt(path.join(hs.image_path, 'TILT_focus_points.txt'), focus_points)
    #focus_points = np.loadtxt(path.join(hs.image_path, 'TILT_focus_points.txt'))

    # Get adjusted z stage positions for level image
    hs.im_obj_pos = 30000
    center, n_ip = planeFit(focus_points[:,0:3])
    n_ip[2] = abs(n_ip[2])
    z_pos = autolevel(hs, n_ip, center)
    np.savetxt(path.join(hs.image_path, 'z_pos_raw.txt'), np.array(z_pos))
    for i in range(3):
        if z_pos[i] > 25000:
            z_pos[i] = 25000
    np.savetxt(path.join(hs.image_path, 'z_pos_check.txt'), np.array(z_pos))
    print('leveling... ', z_pos)
    hs.z.move(z_pos)

    # Move to p=optimal objective position
    for i in range(last_point, len(ord_points)):
        [x_pos, y_pos] = hs.px_to_step(ord_points[i,0], ord_points[i,1], pos_dict, scale)
        hs.y.move(y_pos)
        hs.x.move(x_pos)
        fs = hs.obj_stack()
        f_fs = format_focus(hs, fs)
        if f_fs is not False:
           obj_pos = fit_mixed_gaussian(hs, f_fs)
           if obj_pos:
               break

    # Image leveled planeFit
    hs.y.move(pos_dict['y_initial'])
    hs.x.move(pos_dict['x_initial'])
    hs.obj.move(obj_pos)
    rough_ims, scale = rough_focus(hs, pos_dict['n_tiles'], pos_dict['n_frames'], image_name = 'LVL')
    for i in range(len(hs.channels)):
        imageio.imwrite(path.join(hs.image_path,'c'+str(hs.channels[i])+'LVL.tiff'), rough_ims[i])

    # Get stage positions on in-focus points
    focus_points = get_focus_data(hs, pos_points, n_markers, scale, pos_dict)
    np.savetxt(path.join(hs.image_path, 'LVL_focus_points.txt'), focus_points)

    return z_pos, obj_pos

def rough_focus(hs, n_tiles, n_frames, image_name = 'RoughScan'):
    '''Image section at preset positions and return scaled image scan.'''
    x_initial = hs.x.position
    y_initial = hs.y.position
    #obj_pos = int((hs.obj.focus_stop - hs.obj.focus_start)/2 + hs.obj.focus_start)
    #hs.obj.move(obj_pos)
    # Move to rough focus position
    #z_pos = [hs.z.focus_pos, hs.z.focus_pos, hs.z.focus_pos]
    #hs.z.move(z_pos)
    # Take rough focus image
    hs.scan(n_tiles, 1, n_frames, image_name)
    hs.y.move(y_initial)
    hs.x.move(x_initial)
    rough_ims = []
    # Stitch rough focus image
    for ch in hs.channels:
        df_x = get_image_df(hs.image_path, 'c'+str(ch)+'_'+image_name)
        #df_x = df_x.drop('R', axis = 1)
        #df_x = df_x.rename(columns = {'X0':'ch', 'X2':'x', 'X3':'o'})
        # for i in range(len(df_x.x)):
        #     df_x.x[i] = int(df_x.x[i][1:])
        #     df_x.o[i] = int(df_x.o[i][1:])
        #     df_x.c[i] = int(df_x.c[i][1:])
        plane, scale_factor = stitch(hs.image_path, df_x, scaled = True)
        rough_ims.append(normalize(plane, scale_factor))

    return rough_ims, scale_factor

def find_focus_points(rough_ims, scale, hs):
    x_initial = hs.x.position
    y_initial = hs.y.position
    #scale = rough_ims[0][0][0]
    # Combine channels
    avg_im = image.avg_images(rough_ims)
    # Find region of interest
    roi = image.get_roi(avg_im)
    roi_shape = roi.shape
    roi_f = np.sum(roi)/(roi_shape[0]*roi_shape[1])
    if roi_f >= 0.1:
        # Find focus points
        focus_points = image.get_focus_pos(roi)

        # Shift focus points towards center
        stage_points = np.zeros(shape=[3,2])
        for i in range(1,4):
            focus_point = image.shift_focus(focus_points[i,:],
                                            focus_points[0,:],
                                            2048/2/scale)
            stage_points[i-1,:]= hs.px_to_step(focus_point, x_initial, y_initial,
                                               scale)
    else:
        print('Roi could not be found')

##    # Reorder stage points to match z stage motor indice
##    ordered_stage_points = np.zeros(shape=[3,2])
##
##    m0 = np.where(stage_points[:,0] == np.min(stage_points[:,0]))[0][0]
##    ordered_stage_point[0,:] = stage_points[m0,:]
##    stage_points = np.delete(stage_points,m0,0)
##
##    m1 = np.where(stage_points[:,1] == np.min(stage_points[:,1]))[0][0]
##    ordered_stage_point[1,:] = stage_points[m1,:]
##    stage_points = np.delete(stage_points,m0,0)

    return stage_points

def format_focus2(hs, fs):
    col_names = ['x','y','frame', 'channel', 'filesize', 'kurtosis']

    n_frames, n_channels = fs.shape

    x = np.ones(shape=(n_frames,1))*hs.x.position
    y = np.ones(shape=(n_frames,1))*hs.y.position
    frame = np.array(range(n_frames))
    frame = np.reshape(frame, (n_frames,1))

    for ch in range(n_channels):
        channel = np.ones(shape=(n_frames,1))*hs.channels[ch]
        filesize = np.reshape(fs[:,ch],(n_frames,1))
        kurt = stats.kurtosis(fs)
        kurt = np.ones(shape=(n_frames,1))*kurt
        data = np.concatenate([x, y, frame, channel, filesize, kurt], axis=1)
    df = pd.DataFrame(data, columns = col_names)

    return df

def format_focus(hs, focus_data):
    '''Return valid and normalized focus frame file sizes.

       Parameters:
       - focus (array): JPEG file sizes from N channels of cameras.

       Returns:
       - array: Valid and normalized focus frame file sizes.

    '''
    # Calculate steps per frame
    # hs.obj.v TODO store velocity in mm/s
    #hs.obj.spum = 262 # steps/um
    # frame_interval = hs.cam1.get_interval() # s/frame TODO

##    if len(focus1) != len(focus2):
##        print('Number of focus frame mismatch')
##    else:
##        n_frames = len(focus1)

    if hs.cam1.getFrameInterval() != hs.cam2.getFrameInterval():
        print('Frame interval mismatch')

    frame_interval = hs.cam1.getFrameInterval()
    spf = hs.obj.v*1000*hs.obj.spum*frame_interval # steps/frame

    # Remove frames after objective stops moving
    n_frames = len(focus_data)
    _frames = range(n_frames)
    objsteps = hs.obj.focus_start + np.array(_frames)*spf
    objsteps = objsteps[objsteps < hs.obj.focus_stop]


    # Number of formatted frames
    n_f_frames = len(objsteps)
    objsteps = np.reshape(objsteps, (n_f_frames, 1))

    #formatted focus data
    f_fd = np.zeros(shape = (n_f_frames,1))

    nrows, ncols = focus_data.shape
    for i in range(ncols):
        # test if there is a tail in data and add if True
        kurt_z, pvalue = stats.kurtosistest(focus_data[0:n_f_frames,i])
        if kurt_z > 1.96:
            print('signal in channel ' +str(i))
            f_fd = f_fd + np.reshape(focus_data[0:n_f_frames,i], (n_f_frames,1))

    # Normalize
    if np.sum(f_fd) == 0:
        return False
    else:
        f_fd = f_fd/ np.sum(f_fd)
        f_fd = np.reshape(f_fd, (n_f_frames, 1))

        return np.concatenate((objsteps,f_fd), axis=1)


def gaussian(x, *args):
    '''Gaussian function for curve fitting.'''

    if len(args) == 1:
      args = args[0]

    n_peaks = int(len(args)/3)


    if len(args) - n_peaks*3 != 0:
      print('Unequal number of parameters')
    else:
      for i in range(n_peaks):
        amp = args[0:n_peaks]
        cen = args[n_peaks:n_peaks*2]
        sigma = args[n_peaks*2:n_peaks*3]

      g_sum = 0
      for i in range(len(amp)):
          g_sum += amp[i]*(1/(sigma[i]*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen[i])/sigma[i])**2)))

      return g_sum

def res_gaussian(args, xfun, yfun):
    '''Gaussian residual function for curve fitting.'''

    g_sum = gaussian(xfun, args)

    return yfun-g_sum


def fit_mixed_gaussian(hs, data):
    '''Fit focus data & return optimal objective focus step.

       Focus objective step vs frame JPEG file size is fit to a mixed gaussian
       model. The optimal objective focus step is returned at step of the max
       JPEG file size of the fit.

       Parameters:
       - data (array nx2): Focus data where the 1st column are the objective
                           steps and the 2 column are the valid and normalized
                           focus frame JPEG file size.

       Returns:
       int: The optimal focus objective step. If 1 or -1 is returned, the z
            stage needs to be moved in the +ive or -ive direction to find an
            optimal focus.

    '''

    # initialize values
    max_peaks = 10
    # Initialize varibles
    amp = []; amp_lb = []; amp_ub = []
    cen = []; cen_lb = []; cen_ub = []
    sigma = []; sigma_lb = []; sigma_ub = []
    y = data[:,1]
    SST = np.sum((y-np.mean(y))**2)
    R2 = 0
    tolerance = 0.9                                                             # Tolerance for R2
    xfun = data[:,0]; yfun = data[:,1]

    # Add peaks until fit reaches threshold
    while len(amp) <= max_peaks and R2 < tolerance:
        # set initial guesses
        max_y = np.max(y)
        amp.append(max_y*10000)
        index = np.argmax(y)
        y = np.delete(y, index)
        index = np.where(data[:,1] == max_y)[0][0]
        cen.append(data[index,0])
        sigma.append(np.sum(data[:,1]**2)**0.5*10000)
        p0 = np.array([amp, cen, sigma])
        p0 = p0.flatten()

        # set bounds
        amp_lb.append(0); amp_ub.append(np.inf)
        cen_lb.append(np.min(data[:,0])); cen_ub.append(np.max(data[:,0]))
        sigma_lb.append(0); sigma_ub.append(np.inf)
        lo_bounds = np.array([amp_lb, cen_lb, sigma_lb])
        up_bounds = np.array([amp_ub, cen_ub, sigma_ub])
        lo_bounds = lo_bounds.flatten()
        up_bounds = up_bounds.flatten()

        # Optimize parameters
        results = least_squares(res_gaussian, p0, bounds=(lo_bounds,up_bounds), args=(data[:,0],data[:,1]))

        if not results.success:
            print(results.message)
        else:
            R2 = 1 - np.sum(results.fun**2)/SST
            print('R2 with ' + str(len(amp)) + ' peaks is ' + str(R2))


        if results.success and R2 > tolerance:
            _objsteps = range(hs.obj.focus_start, hs.obj.focus_stop,
                              int(hs.nyquist_obj/2))
            _focus = gaussian(_objsteps, results.x)
            optobjstep = int(_objsteps[np.argmax(_focus)])
            return optobjstep
        else:
            if len(amp) == max_peaks:
                print('No good fit try moving z stage')
                return False

def get_image_df(image_path, image_name = None):
    '''Get dataframe of rough focus images.

    Parameters:
    dir (path): Directory where images are stored.
    image_name (str): Name common to all images.

    Return
    dataframe: Dataframe of image metadata with image names as index.
    '''

    all_names = os.listdir(image_path)
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


# def get_image_df(dir, image_name = None):
#     '''Get dataframe of images.
#
#     Parameters:
#     dir (path): Directory where images are stored.
#     image_name (str): Name common to all images.
#
#     Return
#     dataframe: Dataframe of image metadata with image names as index.
#
#     '''
#
#     all_names = os.listdir(dir)
#     if image_name is None:
#       image_names = [name for name in all_names if '.tiff' in name]
#     else:
#       im_names = [name for name in all_names if image_name in name]
#       image_names = [name for name in im_names if '.tiff' in name]
#
#     # Dataframe for metdata
#     metadata = pd.DataFrame(columns = ('channel','flowcell','specimen',
#                                        'section','cycle','x','o'))
#
#     # Extract metadata
#     for name in image_names:
#
#       meta = name[:-5].split('_')
#
#       # Convert channels to int
#       meta[0] = int(meta[0])
#       # Remove c from cycle
#       meta[4] = int(meta[4][1:])
#       # Remove x from xposition
#       meta[5] = int(meta[5][1:])
#       # Remove o from objective position
#       meta[6] = int(meta[6][1:])
#
#
#       metadata.loc[name] = meta
#
#
#     metadata.sort_values(by=['flowcell', 'specimen', 'section', 'cycle', 'channel',
#                            'o','x'])
#
#     return metadata

def stitch(dir, df_x, overlap = 0, scaled = False):
    '''Stitch together scans.

     Parameters
     dir (path): Directory where image scans are stored.
     df_x (df): Dataframe of metadata of image scans to stitch.
     scaled (bool): True to autoscale images to ~256 Kb, False to not scale.

     Returns
     image: Stitched and scaled image.

    '''

    df_x = df_x.sort_values(by=['x'])
    scale_factor = None
    plane = None
    for name in df_x.index:
        im = io.imread(path.join(dir,name))
        im = im[64:]                                                            # Remove whiteband artifact

        if scaled:
            # Scale images so they are ~ 512 kb
            if scale_factor is None:
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
    '''Normalize scans.

     Images are normalized by matching histogram to strip with most contrast.


     Parameters
     im (array): Image as array
     scale_factor (int): Downscale factor of image

     Returns
     image: Normalized image.

    '''
    x_px = int(2048/scale_factor/8)
    n_strips = int(im.shape[1]/x_px)
    strip_contrast = np.empty(shape=(n_strips,))
    col_contrast = np.max(im, axis = 0) - np.min(im, axis = 0)

    # Find reference strip with max contrast
    for i in range(n_strips):
        strip_contrast[i] = np.mean(col_contrast[(i)*x_px:(i+1)*x_px])
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

    plane = plane.astype('uint8')
    plane = img_as_ubyte(plane)

    return plane

def get_focus_data(hs, px_points, n_markers, scale, pos_dict):
    '''Return points and unit normal that define the imaging plane.

         Loop through candidate focus px_points, and take an objective stack at
         each position until n_markers with an optimal objective position have
         been found.


         Parameters
         px_points (2xN array): Row x Column position of candidate focal points
         n_markers (int): Number of focus markers to find
         scale (int): Scale of image px_points were found

         Returns
         2xn_markers array, 3x1 array: X-stage, Y-stage step of focal position,
                                       Unit normal of imaging plane

    '''
    #pos_dict= {'x_initial':hs.x.position,
    #           'y_initial':hs.y.position,
    #rough_ims, scale = rough_focus(hs, n_tiles, n_frames)
    #sum_im = image.sum_images(rough_ims)
    #n_markers = ((2048*n_tiles)**2 + (n_frames*hs.bundle_height)**2)**0.5   # image size in px
    #n_markers = 3 + int(min_n_markers/2/2048)
    #px_points = image.get_focus_points(im, scale, n_markers*10)
    focus_points = np.empty(shape=(n_markers,4))
    i = 0
    n_obj = 0
    focus_data = None
    while n_obj < n_markers:
    # Take obj_stack at focus points until n_markers have been found
        px_pt = px_points[i,:]
        [x_pos, y_pos] = hs.px_to_step(px_pt[0], px_pt[1], pos_dict, scale)
        hs.y.move(y_pos)
        hs.x.move(x_pos)
        fs = hs.obj_stack()

        # df = format_focus2(hs,fs)
        # if focus_data is None:
        #     focus_data = df
        # else:
        #     focus_data = focus_data.append(df)
        #
        # if any(df.kurtosis > 1.96):
        #     n_obj += 1
        f_fs = format_focus(hs, fs)
        if f_fs is not False:
           obj_pos = fit_mixed_gaussian(hs, f_fs)
           if obj_pos:
               print('Found point ' + str(n_obj))
               ##################################################################
               # Save in focus frame
##               in_focus_frame = np.argmin(abs(f_fs[:,0]-obj_pos))
##               frame_name = '740_'+str(in_focus_frame)+'.jpeg'
##               z_pos = ''
##               for z in z_pos:
##                   z_pos += str(z) + '_'
##               os.rename(path.join(hs.image_path,frame_name), path.join(hs.image_path,
##                         z_pos+str(px_pt[0])+'_'+str(px_pt[1])+'.jpeg'))
               ##################################################################
               # TODO: REMOVE i AFTER TESTING
               focus_points[n_obj,:] = [x_pos, y_pos, obj_pos, i]
               n_obj += 1

        if i == len(px_points)-1:
            n_obj = n_markers+1
            break
        else:
            print(i,'/',len(px_points))
            i += 1


    # Convert stage step position microns
    try:
        focus_points[:,0] = focus_points[:,0]/hs.x.spum
        focus_points[:,1] = focus_points[:,1]/hs.y.spum
        focus_points[:,2] = focus_points[:,2]/hs.obj.spum
    except:
        focus_points = np.array([False])

    #centroid, normal = fit_plane(focus_points)
    #normal[2] = abs(normal[2])

    #return focal_points, normal, centroid
    #return focus_data
    return focus_points

#def get_image_plane(focus_data):


def autolevel(hs, n_ip, centroid):

    # Find normal vector of motor plane
    mp = np.array(hs.z.get_motor_points(),dtype = float)
    mp[:,0] = mp[:,0] / hs.x.spum
    mp[:,1] = mp[:,1] / hs.y.spum
    mp[:,2] = mp[:,2] / hs.z.spum
    u_mp = mp[0,:] - mp[2,:]
    v_mp = mp[1,:] - mp[2,:]
    n_mp = np.cross(u_mp, v_mp)
    n_mp = n_mp/np.linalg.norm(n_mp)
    n_mp[2] = abs(n_mp[2])


    tilt = [0, 0, 1] - n_ip
    # Pick reference motor based on tilt of imaging plane
    if tilt[0] >= 0:
        p_mp = 0 # right motor
    elif tilt[1] >= 0:
        p_mp = 2 # left back motors
    else:
        p_mp = 1 # left front motor

    d_mp = -np.dot(n_mp, mp[p_mp,:])

    # Find objective position on imaging plane at reference motor
    d_ip = -np.dot(n_ip, centroid)


    #obj_mp = -(n_ip[0]*mp[p_mp,0] + n_ip[1]*mp[p_mp,1] + d_ip)/n_ip[2]
    #print(obj_mp)
    #mp[p_mp,2] = obj_mp
    # Calculate plane correction
    #mag_mp = np.linalg.norm(mp[p_mp,:])
    #n_ip = n_ip * mag_mp
    #correction = [0, 0, mag_mp] - n_ip
    correction = [0, 0, 1] - n_ip

    # Calculate target motor plane for a flat, level image plane
    #n_mp = n_mp * mag_mp
    n_tp = n_mp + correction

    #n_tp = n_mp
    n_tp = n_tp / np.linalg.norm(n_tp)
    print('target plane', n_tp)

    # Move motor plane to preferred objective position
    offset_distance = hs.im_obj_pos/hs.obj.spum - centroid[2]
    offset_distance = offset_distance + hs.z.position[p_mp]/hs.z.spum
    mp[p_mp,2] = offset_distance

    #offset_distance = int(offset_distance*hs.z.spum)

    d_tp = -np.dot(n_tp, mp[p_mp,:])

    z_pos = []
    for i in range(3):
        z_pos.append(int(-(n_tp[0]*mp[i,0] + n_tp[1]*mp[i,1] + d_tp)/n_tp[2]))

    # Convert back to z step position
    z_pos = np.array(z_pos)
    z_pos = z_pos * hs.z.spum
    z_pos = z_pos.astype('int')
    #hs.z.move(z_pos)

    return z_pos




def norm_and_stitch(dir, df_x, overlap = 0, scaled = False):
  '''Normalize and stitch scans.

     Images are normalized by matching histogram to first strip in first image.


     Parameters
     dir (path): Directory where image scans are stored.
     df_x (df): Dataframe of metadata of image scans to stitch.
     scaled (bool): True to autoscale images to ~256 Kb, False to not scale.

     Returns
     image: Normalized, stitched, and downscaled image.

  '''

  df_x.sort_values(by=['x'])
  scans = []                                                                    # list of xpos scans
  ref = None
  scale_factor = None
  for name in df_x.index:
    im = io.imread(path.join(dir,name))
    im = im[64:]                                                                # Remove whiteband artifact

    if scaled:
        # Scale images so they are ~ 256 kb
        if scale_factor is None:
            size = stat(path.join(dir,name)).st_size
            scale_factor = (2**(log2(size)-18))**0.5
            scale_factor = round(log2(scale_factor))

        im = downscale_local_mean(im, (scale_factor, scale_factor))

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

  return plane

def planeFit(points):
    '''

    Code copied from https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points

    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    '''

    points = np.transpose(points)

    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:,-1]

# ch = '558' , '610'
# o = 30117
# x = 12344
# scale_factor = 16



# # Dummy class to hold image data
# class image():
#   def __init__(self, data):
#     self.image = data
#     self.elev_map = None
#     self.markers = None
#     self.segmentation = None
#     self.roi = None
#
# # Data frame for images with metadata
# df_imgs_ch = pd.DataFrame(columns = ('channel','flowcell','specimen','section',
#                                      'cycle','o','image'))

# Stitch all images
# In this dataset there are images from cycle 1 from the 10Ab_mouse_4i experiment
# There are 2 channels, 558 nm (GFAP) and 610 nm (IBA1)
# There are 3 objective positions at 28237, 30117, and 31762
# At each objective position there are 4 scans as x pos = 11714, 12029, 12344, and 12659

# for ch in set(metadata.channel):
#   #ch = 558
#   df_ch = metadata[metadata.channel == ch]
#   for o in set(df_ch.o):
#     #o = 30117
#     df_o = df_ch[df_ch.o == o]
#     df_x = df_o.sort_values(by = ['x'])
#     meta = [*df_x.iloc[0]]
#
#     title = 'cy'+str(meta[4])+'_ch'+str(ch)+'_o'+str(o)
#     meta[5] = meta[6]                                                           # Move objective data
#     meta[6] = image(norm_and_stitch(fn, df_x, scale_factor))
#     df_imgs_ch.loc[title] = meta
#
# # Show all images
# i =0
# dim = df_imgs_ch.iloc[0].image.image.shape
# n_images = df_imgs_ch.shape[0]
# if n_images == 1:
#   fig, ax = plt.subplots(figsize=(8, 6))
#   for index, row in df_imgs_ch.iterrows():
#     ax.imshow(row['image'].image, cmap='Spectral')
#     ax.set_title(index)
#     ax.axis('off')
# elif n_images > 1:
#   fig, ax = plt.subplots(1,len(df_imgs_ch), figsize=(dim[0]/10, dim[1]/10*len(df_imgs_ch)))
#   for index, row in df_imgs_ch.iterrows():
#     ax[i].imshow(row['image'].image,cmap='Spectral')
#     ax[i].set_title(index)
#     ax[i].axis('off')
#     i +=1
#
#
#     # Make background 0, assume background is most frequent px value
#     p_back = stats.mode(im, axis=None)
#     imsize = im.shape
#     p_back = p_back[0]
#
#     # Make saturated pixels 0
#     #p_back, p_sat = np.percentile(im, (10,98))
#     p_sat = np.percentile(im, (98,))
#     im[im < p_back] = 0
#     im[im > p_sat] = 0
