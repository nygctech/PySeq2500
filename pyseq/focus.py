#!/usr/bin/python

from . import image

def focus(hs, x_initial, y_initial, n_tiles, n_frames, AorB, section, cycle):
    hs.x.move(x_initial)
    hs.y.move(y_initial)
    hs.obj.move(17500)
    # Move to rough focus position
    hs.z.move([21500, 21500, 21500])
    # Create rough focus image name
    image_name = AorB
    image_name = image_name + '_' + 'R' + str(section)
    image_name = image_name + '_' + 'c' + cycle
    # Take rough focus image
    hs.scan(n_tiles, 1, n_frames, image_name)
    channels = [str(hs.cam1.left_emission),
                str(hs.cam1.right_emission),
                str(hs.cam2.left_emission),
                str(hs.cam2.right_emission)]
    rough_ims = []
    # Stitch rough focus image
    for ch in channels:
        df_x = image.get_image_df(hs.image_path, ch+'_'+image_name)
        rough_ims.append(norm_and_stitch(hs.image_path, df_x, scaled = True))
    scale = rough_ims[0][0][0]
    # Combine channels
    avg_im = image.average_images(rough_ims)
    # Find region of interest
    roi = image.get_roi(avg_im)
    # Find focus points
    focus_points = get_focus_points(roi)
    # Shift focus points towards center
    stage_points = []
    for i in range(1,4):
        focus_point = image.shift_focus(focus_points[i,:],
                                        focus_points[0,:],
                                        2048/2/scale)
        stage_points.append(hs.px_to_step(focus_point, x_initial, y_initial,
                                          scale))
    print(stage_points)


def get_image_df(dir, image_name = None):
  '''Get dataframe of images.

    Parameters:
    dir (path): Directory where images are stored.
    image_name (str): Name common to all images.

    Return
    dataframe: Dataframe of image metadata with image names as index.

  '''

    all_names = os.listdir(dir)
    if image_name is None:
      image_names = all_names
    else:
      image_names = [name for name in all_names if image_name in name]

    # Dataframe for metdata
    metadata = pd.DataFrame(columns = ('channel','flowcell','specimen',
                                       'section','cycle','x','o'))

    # Extract metadata
    for name in image_names:

      meta = name[:-5].split('_')

      # Convert channels to int
      meta[0] = int(meta[0])
      # Remove c from cycle
      meta[4] = int(meta[4][1:])
      # Remove x from xposition
      meta[5] = int(meta[5][1:])
      # Remove o from objective position
      meta[6] = int(meta[6][1:])


      metadata.loc[name] = meta


  metadata.sort_values(by=['flowcell', 'specimen', 'section', 'cycle', 'channel',
                           'o','x'])

  return metadata



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

    if scaled = True:
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
        sub_im = exposure.match_histograms(sub_im, ref)
        plane = np.append(plane, sub_im, axis = 1)

  plane = plane.astype('uint8')
  plane = img_as_ubyte(plane)

  return plane

ch = '558' , '610'
o = 30117
x = 12344
scale_factor = 16



# Dummy class to hold image data
class image():
  def __init__(self, data):
    self.image = data
    self.elev_map = None
    self.markers = None
    self.segmentation = None
    self.roi = None

# Data frame for images with metadata
df_imgs_ch = pd.DataFrame(columns = ('channel','flowcell','specimen','section',
                                     'cycle','o','image'))

# Stitch all images
# In this dataset there are images from cycle 1 from the 10Ab_mouse_4i experiment
# There are 2 channels, 558 nm (GFAP) and 610 nm (IBA1)
# There are 3 objective positions at 28237, 30117, and 31762
# At each objective position there are 4 scans as x pos = 11714, 12029, 12344, and 12659

for ch in set(metadata.channel):
  #ch = 558
  df_ch = metadata[metadata.channel == ch]
  for o in set(df_ch.o):
    #o = 30117
    df_o = df_ch[df_ch.o == o]
    df_x = df_o.sort_values(by = ['x'])
    meta = [*df_x.iloc[0]]

    title = 'cy'+str(meta[4])+'_ch'+str(ch)+'_o'+str(o)
    meta[5] = meta[6]                                                           # Move objective data
    meta[6] = image(norm_and_stitch(fn, df_x, scale_factor))
    df_imgs_ch.loc[title] = meta

# Show all images
i =0
dim = df_imgs_ch.iloc[0].image.image.shape
n_images = df_imgs_ch.shape[0]
if n_images == 1:
  fig, ax = plt.subplots(figsize=(8, 6))
  for index, row in df_imgs_ch.iterrows():
    ax.imshow(row['image'].image, cmap='Spectral')
    ax.set_title(index)
    ax.axis('off')
elif n_images > 1:
  fig, ax = plt.subplots(1,len(df_imgs_ch), figsize=(dim[0]/10, dim[1]/10*len(df_imgs_ch)))
  for index, row in df_imgs_ch.iterrows():
    ax[i].imshow(row['image'].image,cmap='Spectral')
    ax[i].set_title(index)
    ax[i].axis('off')
    i +=1


    # Make background 0, assume background is most frequent px value
    p_back = stats.mode(im, axis=None)
    imsize = im.shape
    p_back = p_back[0]

    # Make saturated pixels 0
    #p_back, p_sat = np.percentile(im, (10,98))
    p_sat = np.percentile(im, (98,))
    im[im < p_back] = 0
    im[im > p_sat] = 0
