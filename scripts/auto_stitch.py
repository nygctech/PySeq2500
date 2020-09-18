import os
import subprocess
from os.path import join
import numpy as np
from pyseq import image_analysis as ia

###CHANGE ME###

path = 'Y:\\Kunal\\HiSeqExperiments\\filter_optimization\\Cy3+Cy5+AF700'



#########

output = subprocess.run(['findstr', 'imaging completed', log], stdout=subpro
cess.PIPE).stdout.decode('utf-8')
section = 5
cycle = 1

df = ia.get_image_df(path, images)

def get_image_info(path):
    dir_list = os.listdir(path)
    img_list = []
    channels = []
    flowcells = []
    sections = []
    cycles = []
    x = []
    o = []
    for i in dir_list:
        if i.endswith('tiff'):
            
            img_list.append(i)
            
            img_name = np.array(i.split('_'))
            n = np.array(range(len(img_name)))
            
            if img_name[0] not in channels:
                channels.append(img_name[0])
                
            if img_name[1] not in flowcells:
                flowcells.append(img_name[1])

            section_name = img_name[2] + '_' + img_name[3]    
            if section_name not in sections:
                sections.append(section_name)

            if img_name[4] not in cycles:
                cycles.append(img_name[4])

            if img_name[5] not in x:
                x.append(img_name[5])

            obj_pos = img_name[6].split('.')[0]
            if obj_pos not in o:
                o.append(obj_pos)

    x.sort()
    
    return img_list, channels, flowcells, sections, cycles, x, o


path = path + 'section' + str(section) + '\\cycle'+str(cycle)+'\\'
img_list, channels, flowcells, sections, cycles, xpos, opos = get_image_info(path)

print(channels)
print(flowcells)
print(sections)
print(cycles)
print(xpos)
print(opos)


crop = False
preview = True

crop_width = 6516
crop_height = 6936
crop_x = 2920
crop_y = 804

if preview:
    opos = [opos[0]]
    crop = False


for f in flowcells:
    for s in sections:
        for cy in cycles:
            for o in opos:
                for ch in channels:
                    
                    command = 'magick '
                    command += 'montage -tile x1 -geometry +0+0 '
                    
                    for x in xpos:
                        img_name = ch+'_'+f+'_'+s+'_'+cy+'_'+x+'_'+o+'.tiff'
                        command += path + img_name + ' '

                    if crop:
                        command += path + 'temp.tiff'
                        os.system(command)
                        
                        command = 'magick '
                        command += path + 'temp.tiff '
                        command +=  '-crop ' + str(crop_width) + 'x' + str(crop_height)
                        command += '+' + str(crop_x) + '+' + str(crop_y) + ' '
                        
                    stitch_name = ch+'_'+f+'_'+s+'_'+cy+'_'+o+'.tiff'
                    command += path + stitch_name
                    os.system(command)

                        
                        
        
