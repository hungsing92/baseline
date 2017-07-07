from net.common import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from time import time

import cv2
from net.utility.draw import *
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================




## lidar to top ##
def lidar_to_top(lidar):

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    width  = Yn - Y0
    height   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    filter_x=np.where((pxs>=TOP_X_MIN) & (pxs<=TOP_X_MAX))[0]
    filter_y=np.where((pys>=TOP_Y_MIN) & (pys<=TOP_Y_MAX))[0]
    filter_z=np.where((pzs>=TOP_Z_MIN) & (pzs<=TOP_Z_MAX))[0]
    filter_xy=np.intersect1d(filter_x,filter_y)
    filter_xyz=np.intersect1d(filter_xy,filter_z)
    pxs=pxs[filter_xyz]
    pys=pys[filter_xyz]
    pzs=pzs[filter_xyz]
    prs=prs[filter_xyz]   

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    # top[:,:,-2:]=0
    mask = np.ones(shape=(height,width,channel-1), dtype=np.float32)* -5
    # pdb.set_trace()

    for i in range(len(pxs)):
        # top[-qxs[i], -qys[i], qzs[i]] = qzs[i]*TOP_Z_DIVISION+TOP_Z_MIN
        top[-qxs[i], -qys[i], -1]= 1+ top[-qxs[i], -qys[i], -1]
        if pzs[i]>mask[-qxs[i], -qys[i],qzs[i]]:
            top[-qxs[i], -qys[i], qzs[i]] = max(0,pzs[i]-TOP_Z_MIN)
            mask[-qxs[i], -qys[i],qzs[i]]=pzs[i]
        if pzs[i]>mask[-qxs[i], -qys[i],-1]:
            mask[-qxs[i], -qys[i],-1]=pzs[i]
            top[-qxs[i], -qys[i], -2]=prs[i]






#     ## start to make top  here !!!
#     start=time()
#     for z in range(Z0,Zn):
#         iz = np.where (qzs==z)
#         for y in range(Y0,Yn):
#             iy  = np.where (qys==y)
#             iyz = np.intersect1d(iy, iz)

#             for x in range(X0,Xn):
#                 #print('', end='\r',flush=True)
#                 #print(z,y,z,flush=True)

#                 ix = np.where (qxs==x)
#                 idx = np.intersect1d(ix,iyz)

#                 if len(idx)>0:
#                     yy,xx,zz = -(x-X0),-(y-Y0),z-Z0

#                     #height per slice
#                     max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
#                     top[yy,xx,zz]=max_height

#                     #intensity
#                     max_intensity = np.max(prs[idx])
#                     top[yy,xx,Zn]=max_intensity

#                     #density
#                     count = len(idx)
#                     top[yy,xx,Zn+1]+=count
# w
#                 pass
#             pass
#         pass

#     print("speed:%fs"%(time()-start))
    # top[:,:,:-2]=top[:,:,:-2]-TOP_Z_MIN
    # top[top[:,:,:-2]<0]=0
    top[:,:,-1] = np.log(top[:,:,-1]+1)/math.log(64)

    if 1:
        # top_image = np.sum(top[:,:,:-1],axis=2)
        density_image=top[:,:,-1]
        density_image = density_image-np.min(density_image)
        density_image = (density_image/np.max(density_image)*255).astype(np.uint8)
        # top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


#     if 0: #unprocess
#         top_image = np.zeros((height,width,3),dtype=np.float32)

#         num = len(lidar)
#         for n in range(num):
#             x,y = qxs[n],qys[n]
#             if x>=0 and x <width and y>0 and y<height:
#                 top_image[y,x,:] += 1

#         max_value=np.max(np.log(top_image+0.001))
#         top_image = top_image/max_value *255
#         top_image=top_image.astype(dtype=np.uint8)

    return top, density_image

root_dir = "/home/hhs/4T/datasets/KITTI/object/training"
velodyne = os.path.join(root_dir, "velodyne/")
bird = os.path.join(root_dir, "lidar_bv/")


for i in range(7481):
    # i=i+7253
    filename = velodyne + str(i).zfill(6) + ".bin"
    print("Processing: ", filename)
    lidar = np.fromfile(filename, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    # pdb.set_trace()
    top_new, density_image=lidar_to_top(lidar)
    # img_=lidar_new[:,:,-1]
    # img_ = img_-np.min(img_)
    # img_ = (img_/np.max(img_)*255)
    # imshow('bv', top_image)
    # imshow('density', img_)

    # lidar_o= tf.placeholder(shape=lidar.shape, dtype=tf.float32, name='lidar'  )
    # tops, top_images =tf.py_func(lidar_to_top,[lidar_o],[tf.float32, tf.uint8])
    # sess=tf.InteractiveSession()
    # sess.run( tf.global_variables_initializer())
    # with sess.as_default(): 
    #     fd={lidar_o:lidar}
    #     top,top_image=sess.run([tops,top_images],fd)
    # np.save('/home/hhs/4T/datasets/dummy_datas/seg/lidar/lidar_%05d.npy'%i,lidar)
    np.save('/home/hhs/4T/datasets/dummy_datas/seg/top_7/top_70%05d.npy'%i,top_new)
    cv2.imwrite('/home/hhs/4T/datasets/dummy_datas/seg/density_image_7/density_image_70%05d.png'%i,density_image)
   
   
    
    
    
#
# # test
# test = np.load(bird + "000008.npy")

# print(test.shape)
# plt.imshow(test[:,:,8])
# plt.show()



