#!/usr/bin/python

import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import xarray as xr
xr.set_options(keep_attrs=True)
import zarr
import napari
from math import log2, ceil, floor
from os import listdir, stat, path, getcwd, mkdir
from scipy import stats
from scipy.spatial.distance import cdist
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



def sum_images(images, logger = None):
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
        k = kurt(images.sel(channel=ch))
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
    im = im.values
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
    im = im.values
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
      markers = markers[rand_markers]
      n_markers = 1000



    # Compute contrast
    c_score = np.zeros_like(markers)
    for row in range(n_markers):
      frame = im[markers[row]]
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
        - viewer (napari): Napari viewer
        - logger: Logger object to log communication with HiSeq and user.
        - filenames: Files used to stitch image


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
        self.filenames = []
        self.resolution = 0.375                                                 # um/px
        self.x_spum = 0.4096                                                    #steps per um

        if len(common_name) > 0:
            common_name = '*'+common_name

        section_names = []
        if im is None:
            if image_path is None:
                image_path = getcwd()

            # Open zarr
            if image_path[-4:] == 'zarr':
                self.filenames = [image_path]
                section_names = self.open_zarr()

            if obj_stack is not None:
                # Open obj stack (jpegs)
                #filenames = glob.glob(path.join(image_path, common_name+'*.jpeg'))
                n_frames = self.open_objstack(obj_stack)

            elif RoughScan:
                # RoughScans
                filenames = glob.glob(path.join(image_path,'*RoughScan*.tiff'))
                if len(filenames) > 0:
                    self.filenames = filenames
                    n_tiles = self.open_RoughScan()

            else:
                # Open tiffs
                filenames = glob.glob(path.join(image_path, common_name+'*.tiff'))
                if len(filenames) > 0:
                    self.filenames = filenames
                    section_names += self.open_tiffs()

                # Open zarrs
                filenames = glob.glob(path.join(image_path, common_name+'*.zarr'))
                if len(filenames) > 0:
                    self.filenames = filenames
                    section_names += self.open_zarr()

            if len(section_names) > 0:
                message(self.logger, 'Opened', *section_names)

        else:
            self.im = im


    def correct_background(self):

        if not bool(self.im.fixed_bg):
            # Open background config
            config = configparser.ConfigParser()
            if not self.im.machine:
                config_path = pkg_resources.path(resources, 'background.cfg')
            else:
                config_path = pkg_resources.path(resources, machine+'.cfg')
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
            self.im.attrs['fixed_bg'] = 1
        else:
            print('Image already background corrected.')


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



    def remove_overlap(self, overlap=0, direction = 'left'):
        """Remove pixel overlap between tile."""

        try:
            overlap=int(overlap)
            n_tiles = int(len(self.im.col)/2048)
        except:
            print('overlap must be an integer')

        try:
            if direction.lower() in ['l','le','lef','left','lft','lt']:
                direction = 'left'
            elif direction.lower() in ['r','ri','riht','right','rht','rt']:
                direction = 'right'
            else:
                raise ValueError
        except:
            print('overlap direction must be either left or right')

        if not bool(self.im.overlap):
            if n_tiles > 1 and overlap > 0:
                tiles = []
                for t in range(n_tiles):
                    if direction == 'left':
                        cols = slice(2048*t+overlap,2048*(t+1))                 #initial columns are cropped from subsequent tiles
                        tiles.append(self.im.sel(col=cols))
                    elif direction == 'right':
                        cols = slice(2048*t,(2048-overlap)*(t+1))               #end columns are cropped from subsequent tiles
                        tiles.append(self.im.sel(col=cols))
                im = xr.concat(tiles, dim = 'col')
                im.attrs['overlap'] = overlap

                self.im = im
        else:
            print('Overlap already removed')



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
            old_scale = self.im.attrs['scale']
            self.im.attrs['scale']=scale*old_scale


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


    def save_zarr(self, save_path, show_progress = True):
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

        if show_progress:
            with ProgressBar() as pbar:
                self.im.to_dataset().to_zarr(save_name)
        else:
            self.im.to_dataset().to_zarr(save_name)


        # save attributes
        f = open(path.join(save_path, self.im.name+'.attrs'),"w")
        for key, val in self.im.attrs.items():
            f.write(str(key)+' '+str(val)+'\n')
        f.close()

    def open_zarr(self):
        """Create labeled dataset from zarrs.

           **Parameters:**
           - filename(list): List of full file path names to images

           **Returns:**
           - array: Labeled dataset

        """

        im_names = []
        for fn in self.filenames:
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


    def open_RoughScan(self):
        # Open RoughScan tiffs
        filenames = self.filenames

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

        im = im.assign_attrs(first_group = 0, machine = '', scale=1, overlap=0,
                             fixed_bg = 0)
        self.im = im.sel(row=slice(64,None))

        return len(fn_comp_sets[2])

    def open_objstack(self, obj_stack):

        dim_names = ['frame', 'channel', 'row', 'col']
        channels = [687, 558, 610, 740]
        frames = range(obj_stack.shape[0])
        coord_values = {'channel':channels, 'frame':frames}
        im = xr.DataArray(obj_stack.tolist(),
                               dims = dim_names,
                               coords = coord_values,
                               name = 'Objective Stack')

        im = im.assign_attrs(first_group = 0, machine = '', scale=1,
                             overlap=0, fixed_bg = 0)
        self.im = im

        return obj_stack.shape[0]


    def open_tiffs(self):
        """Create labeled dataset from tiffs.

           **Parameters:**
           - filename(list): List of full file path names to images

           **Returns:**
           - array: Labeled dataset

        """

        # Open tiffs
        filenames = self.filenames
        section_sets = dict()
        section_meta = dict()
        for fn in filenames:
            # Break up filename into components
            comp_ = path.basename(fn)[:-5].split("_")
            if len(comp_) >= 6:
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
            fn_comp_sets = list(section_sets[s].values())
            if len(comp_) == 6:
                comp_order = {'ch':0, 'AorB':1, 's':2, 'r':3, 'x':4, 'o':5}
            elif len(comp_) == 7:
                comp_order = {'ch':0, 'AorB':1, 's':2, 'r':3, 'i':4, 'x':5, 'o':6}
            int_comps = ['ch', 'r', 'x', 'o']
            for i in [comp_order[c] for c in comp_order.keys() if c in int_comps]:
                fn_comp_sets[i] = [int(x[1:]) for x in fn_comp_sets[i]]
                fn_comp_sets[i] = sorted(fn_comp_sets[i])
            if 'i' in comp_order.keys():
                i = comp_order['i']
                fn_comp_sets[i] = [int(x) for x in fn_comp_sets[i]]
                fn_comp_sets[i] = sorted(fn_comp_sets[i])
                remap_comps = [fn_comp_sets[0], fn_comp_sets[3], fn_comp_sets[4], fn_comp_sets[6], [1],  fn_comp_sets[5]]
                # List of sorted x steps for calculating overlap
                #x_steps = sorted(list(fn_comp_sets[5]), reverse=True)
                x_steps = fn_comp_sets[5]
            else:
                remap_comps = [fn_comp_sets[0], fn_comp_sets[3], fn_comp_sets[5], [1],  fn_comp_sets[4]]
                # List of sorted x steps for calculating overlap
                #x_steps = sorted(list(fn_comp_sets[4]), reverse=True)
                x_steps = fn_comp_sets[4]

            a = np.empty(tuple(map(len, remap_comps)), dtype=object)
            for fn, x in zip(filenames, lazy_arrays):
                comp_ = path.basename(fn)[:-5].split("_")
                channel = fn_comp_sets[0].index(int(comp_[0][1:]))
                cycle = fn_comp_sets[3].index(int(comp_[3][1:]))
                co = comp_order['o']
                obj_step = fn_comp_sets[co].index(int(comp_[co][1:]))
                co = comp_order['x']
                x_step = fn_comp_sets[co].index(int(comp_[co][1:]))
                if 'i' in comp_order.keys():
                    co = comp_order['i']
                    image_i = fn_comp_sets[co].index(int(comp_[co]))
                    a[channel, cycle, image_i, obj_step, 0, x_step] = x
                else:
                    a[channel, cycle, obj_step, 0, x_step] = x

            # Label array
            if 'i' in comp_order.keys():
                dim_names = ['channel', 'cycle', 'image', 'obj_step', 'row', 'col']
                coord_values = {'channel':fn_comp_sets[0], 'cycle':fn_comp_sets[3], 'image':fn_comp_sets[4], 'obj_step':fn_comp_sets[6]}
            else:
                dim_names = ['channel', 'cycle', 'obj_step', 'row', 'col']
                coord_values = {'channel':fn_comp_sets[0], 'cycle':fn_comp_sets[3], 'obj_step':fn_comp_sets[5]}
            im = xr.DataArray(da.block(a.tolist()),
                                   dims = dim_names,
                                   coords = coord_values,
                                   name = s[1:])


            im = self.register_channels(im.squeeze())
            im = im.assign_attrs(first_group = 0, machine = '', scale=1,
                                 overlap=0, fixed_bg = 0)
            self.im.append(im)
            im_names.append(s[1:])

        return im_names
