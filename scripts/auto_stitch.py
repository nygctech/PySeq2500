#!/usr/bin/env python
import sys
sys.path.append('C:\\Users\\kpandit\\PySeq2500\\pyseq')
import image_analysis as ia

# import image_analysis as ia

import subprocess
import time
import os
from os.path import join, exists, basename
from os import mkdir, getcwd
import imageio


def wait_for_new_images(log_path, im_path, n_old_images, sleep_time = 100):
    if os.name is 'posix':
        im_command = ['grep', '-e', 'Imaging completed', log_path]
        end_command =  ['grep', '-e', 'PySeq::Shutting down', log_path]

    else:
        im_command = ['findstr', 'Imaging completed', log_path]
        end_command =  ['findstr', 'PySeq::Shutting down', log_path]

    ftime = time.asctime(time.localtime(time.time()))
    print(ftime,' - Waiting for new images')
    new_images = False
    while not new_images:
        output = subprocess.run(im_command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        image_count = output.count('\r\n')
        if image_count > n_old_images:
            new_images = True
        else:
            time.sleep(sleep_time)

        complete = subprocess.run(end_command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        if len(complete) > 0:
            break

    new = ia.get_image_df(im_path)

    return new, image_count

# Compensation
##comp = {558: False,
##        610: {'m':1.01, 'b':6.67, 'c':558},
##        687: False,
##        740: {'m':1.17, 'b':-5.7, 'c':687}
##        }
def main(compensation = False):
    exp_path = getcwd()
    name = basename(exp_path)
    print('Autostitching', name)
    log_path = join(exp_path,'logs',name+'.log')

    im_path = join(exp_path, 'images')
    stitch_path = join(exp_path, 'stitched')
    thumb_path = join(exp_path, 'thumbnails')
    if not exists(stitch_path):
        mkdir(stitch_path)
    if not exists(thumb_path):
        mkdir(thumb_path)

    n_old_images = 0
    new, n_old_images = wait_for_new_images(log_path, im_path, n_old_images)
    old =  new.iloc[0:0,:].copy()
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
                        ftime = time.asctime(time.localtime(time.time()))
                        im_name = 'c'+str(c)+'_'+s+'_r'+str(r)+'.tiff'
                        print(ftime, ' - Making ' + im_name)
                        plane = ia.make_image(im_path, df_x, compensation)
                        imageio.imwrite(join(stitch_path, im_name), plane)
                        # Make thumbnails
                        plane, scale = ia.stitch(im_path, df_x, scaled = True)
                        plane = ia.normalize(plane, scale)
                        im_name += '_f'+str(scale)+'.tiff'
                        imageio.imwrite(join(thumb_path, im_name), plane)
        old = old.append(new)
        new, n_old_images = wait_for_new_images(log_path, im_path, n_old_images, sleep_time = 10)
        new = new[new.eq(old).all(axis=1)==False]



if __name__ == '__main__':
    main()
