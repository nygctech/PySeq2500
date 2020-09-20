#!/usr/bin/env python

#from pyseq import image_analysis as ia
import sys
sys.path.append('C:\\Users\\kpandit\\PySeq2500\\pyseq\\')
import image_analysis as ia

import subprocess
import time
from os.path import join, exists, basename
from os import mkdir
import imageio


if __name__ == '__main__':
    main()



def wait_for_new_images(log_path, sleep_time = 100):
    new_images = False
    while not new_images:
        output = subprocess.run(['findstr', 'imaging completed', log_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if output.count('\r\n') > 0:
            new_image = True
        else:
            time.sleep(sleep_time)

        complete = subprocess.run(['findstr', 'PySeq::Shutting down', log_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if len(complete) > 0:
            break

    new = ia.get_image_df(im_path)

    return new

# Compensation
##comp = {558: False,
##        610: {'m':1.01, 'b':6.67, 'c':558},
##        687: False,
##        740: {'m':1.17, 'b':-5.7, 'c':687}
##        }
def main(compensation = False):
    exp_dir = os.getcwd()
    comp = False

    log_path = join(exp_dir,'logs',name+'.log')
    name = basename(exp_dir)
    im_path = join(exp_path, 'images')
    stitch_path = join(exp_path, 'stitched')
    thumb_path = join(exp_path, 'thumbnails')
    if not exists(stitch_path):
        mkdir(stitch_path)
    if not exists(thumb_path):
        mkdir(thumb_path)

    new = wait_for_new_images(log_path)
    while len(new)>0:
        for s in set(new.s):
            df_s = new[new.s == s]
            for r in set(df_s.r):
                df_r = df_s[df_s.r == r]
                for o in set(df_r.o):
                    df_o = df_r[df_r.o == o]
                    for c in set(df_o.c):
                        df_x = df_o[df_o.c == c]
                        # Make full resolution images
                        im_name = 'c'+str(c)+'_'+s+'_r'+str(r)+'.tiff'
                        print('Making ' + im_name)
                        plane = ia.make_image(im_path, df_x, comp)
                        imageio.imwrite(join(stitch_path, im_name), plane)
                        # Make thumbnails
                        plane, scale = ia.stitch(im_path, df_x, True)
                        plane = ia.normalize(plane, scale)
                        im_name = 'c'+str(c)+'_'+s+'_r'+str(r)+'_f'+str(scale)+'.tiff'
                        imageio.imwrite(join(thumb_path, im_name), plane)

        old = new
        new = wait_for_new_images(log_path)
        new = old[old.eq(new).all(axis=1)==False]
