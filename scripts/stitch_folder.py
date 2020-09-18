#from pyseq import image_analysis as ia
import sys
sys.path.append('C:\\Users\\kpandit\\PySeq2500\\pyseq\\')

import subprocess
import time
import image_analysis as ia
from os.path import join, exists
from os import mkdir
import imageio

###CHANGE ME###
name = 'Cy3+Cy5+AF700'
path = 'Y:\\Kunal\\HiSeqExperiments\\filter_optimization\\'
scaled = False
comp = False
###############

path = join(path, name)
log_path = join(path,'logs',name+'.log')
im_path = join(path, 'images')
save_path = join(path, 'stitched')
if not exists(save_path):
    mkdir(save_path)

new_images = False
while not new_images:
    output = subprocess.run(['findstr', 'imaging completed', log_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if output.count('\r\n') > 0:
        new_image = True
    else:
        time.sleep(100)
        
new = ia.get_image_df(im_path)

# Compensation
##comp = {558: False,
##        610: {'m':1.01, 'b':6.67, 'c':558},
##        687: False,
##        740: {'m':1.17, 'b':-5.7, 'c':687}
##        }

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

    old = new
    new = ia.get_image_df(im_path)
    new = old[old.eq(new).all(axis=1)==False]
    #new = df[df.ne(df_old).all(axis=1) == True]
