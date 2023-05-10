import cv2
import json
import numpy as np
from glob import glob

imgs_files = glob('bbox_imgs/*')
annotated_files = glob('gt_images/*')

i=0
src_img_names = [im.split('/')[-1].split('.')[0] for im in imgs_files]
dist_img_names = [im.split('/')[-1][:-4] for im in annotated_files]
annotated_file = open('annotated_imgs.txt','r+')
annotated_names = annotated_file.readlines()
annotated_names = [name.split('\n')[0] for name in annotated_names]

while(i < len(imgs_files)):

    im_name = imgs_files[i].split('/')[-1].split('.')[0]
    if im_name in annotated_names:
        i+=1
        continue
        
    if im_name in dist_img_names:
        i+=1
        continue
    annotated_file.write(im_name+'\n')
    img = cv2.imread(imgs_files[i])
    noised_bg = cv2.resize(img,(150,250))
    # noised_bg = cv2.blur(noised_bg,(25,25))
    noised= cv2.cvtColor(noised_bg,cv2.COLOR_BGR2GRAY)
    noised_bg[:,:,0]=noised
    # noised_bg[:,:,1]= np.random.randint(0,255,[noised.shape[0],noised.shape[1]])
    # noised_bg[:,:,2] = np.random.randint(0,255,[noised.shape[0],noised.shape[1]])
    cv2.imshow('imfile',noised_bg)
    is_gender=False
    is_clss = False
    gender='u'
    clss=0
    while( not is_gender or not is_clss):
        key = cv2.waitKeyEx()

        if key==102:
            gender='f'
            is_gender=True
        elif key ==109:
            gender='m'
            is_gender=True
        elif key==120:
            gender='x'
            is_gender=True
        if key==49:
            clss='p'
            is_clss=True
        elif key==50:
            clss='s'
            is_clss=True
        elif key==51:
            clss='h'
            is_clss=True
        elif key==52:
            clss='d'
            is_clss=True
        elif key==53:
            clss='n'
            is_clss=True
        if key==63234:
            i-=2
            break
        if key ==54:
            exit()
        if key == 55:
            break
            
    if is_gender and is_clss:

        img_name = imgs_files[i].split('/')[-1].split('.')[0]
        new_img_name = img_name +'_'+gender+'_'+str(clss)+'.jpg'
        cv2.imwrite('gt_imgs/'+new_img_name,img)
    i+=1





        


#@ keeping noting while the meeting

#@ having a look at the clender in the morning 






#right 63235
# lift 63234
#f -> 102
#m ->109
#49->1 porn
#50->2 sexy
#51->3 hintai
#52->4 drawing
#53->5 natural
