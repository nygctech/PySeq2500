from pyseq import image_analysis as ia
from os.path import join, exists
from os import mkdir
import imageio

path = '/Volumes/Faculty/Innovation Lab/Kunal/HiSeqExperiments/filter_optimization/Cy3+Cy5+AF700'
im_path = join(path, 'images')
save_path = join(im_path, 'stitched')
if not exists(save_path):
    mkdir(save_path)

new = ia.get_image_df(im_path)
#new = new.drop(index = new[new.s == 'ss11m'].index)

# Compensation
comp = {558: False,
        610: {'m':1.01, 'b':6.67, 'c':558},
        687: False,
        740: {'m':1.17, 'b':-5.7, 'c':687}
        }

scaled = False
while len(new)>0:
    for s in set(new.s):
        df_s = new[new.s == s]
        for r in set(df_s.r):
            df_r = df_s[df_s.r == r]
            for o in set(df_r.o):
                df_o = df_r[df_r.o == o]
                for c in set(df_o.c):
                    df_x = df_o[df_o.c == c]
                    if scaled:
                        plane, scale = ia.stitch(im_path, df_x, scaled)
                        plane = ia.normalize(plane, scale)
                        im_name = 'c'+str(c)+'_'+s+'_r'+str(r)+'_f'+str(scale)+'.tiff'
                    else:
                        im_name = 'c'+str(c)+'_'+s+'_r'+str(r)+'.tiff'
                        print('Making ' + im_name)
                        plane = ia.make_image(im_path, df_x, comp)

                    imageio.imwrite(join(save_path, im_name), plane)

    df_old = new
    df = ia.get_image_df(im_path)
    new = []
    #new = df[df.ne(df_old).all(axis=1) == True]
