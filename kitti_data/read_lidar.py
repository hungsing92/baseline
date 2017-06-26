from net.common import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from time import time
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
## lidar to top ##
def lidar_to_top(lidar):

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)//TOP_Z_DIVISION)+1
    height  = Yn - Y0
    width   = Xn - X0
    channel = Zn - Z0  + 2

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)

    print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)
    # pdb.set_trace()

    filter_x=np.where((pxs>=TOP_X_MIN) & (pxs<=TOP_X_MAX))[0]
    filter_y=np.where((pys>=TOP_Y_MIN) & (pys<=TOP_Y_MAX))[0]
    filter_z=np.where((pzs>=TOP_Z_MIN) & (pzs<=TOP_Z_MAX))[0]
    filter_xy=np.intersect1d(filter_x,filter_y)
    filter_xyz=np.intersect1d(filter_xy,filter_z)
    pxs=pxs[filter_xyz]
    pys=pys[filter_xyz]
    pzs=pzs[filter_xyz]
    prs=prs[filter_xyz]   
    qxs=qxs[filter_xyz]
    qys=qys[filter_xyz]
    qzs=qzs[filter_xyz]

    ## start to make top  here !!!
    start=time()
    for z in range(Z0,Zn):
        iz = np.where (qzs==z)
        for y in range(Y0,Yn):
            iy  = np.where (qys==y)
            iyz = np.intersect1d(iy, iz)

            for x in range(X0,Xn):
                #print('', end='\r',flush=True)
                #print(z,y,z,flush=True)

                ix = np.where (qxs==x)
                idx = np.intersect1d(ix,iyz)

                if len(idx)>0:
                    yy,xx,zz = -(x-X0),-(y-Y0),z-Z0

                    #height per slice
                    max_height = max(0,np.max(pzs[idx])-TOP_Z_MIN)
                    top[yy,xx,zz]=max_height

                    #intensity
                    max_intensity = np.max(prs[idx])
                    top[yy,xx,Zn]=max_intensity

                    #density
                    count = len(idx)
                    top[yy,xx,Zn+1]+=count
w
                pass
            pass
        pass
    print("speed:%fs"%(time()-start))

    top[:,:,Zn+1] = np.log(top[:,:,Zn+1]+1)/math.log(64)

    if 1:
        top_image = np.sum(top,axis=2)
        top_image = top_image-np.min(top_image)
        top_image = (top_image/np.max(top_image)*255)
        top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)


    if 0: #unprocess
        top_image = np.zeros((height,width,3),dtype=np.float32)

        num = len(lidar)
        for n in range(num):
            x,y = qxs[n],qys[n]
            if x>=0 and x <width and y>0 and y<height:
                top_image[y,x,:] += 1

        max_value=np.max(np.log(top_image+0.001))
        top_image = top_image/max_value *255
        top_image=top_image.astype(dtype=np.uint8)

    return top, top_image

root_dir = "/home/hhs/4T/datasets/KITTI/object/training"
velodyne = os.path.join(root_dir, "velodyne/")
bird = os.path.join(root_dir, "lidar_bv/")


for i in range(300):
    i=i+7253
    filename = velodyne + str(i).zfill(6) + ".bin"
    print("Processing: ", filename)
    lidar = np.fromfile(filename, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))
    lidar_o= tf.placeholder(shape=lidar.shape, dtype=tf.float32, name='lidar'  )
    tops, top_images =tf.py_func(lidar_to_top,[lidar_o],[tf.float32, tf.uint8])
    sess=tf.InteractiveSession()
    sess.run( tf.global_variables_initializer())
    with sess.as_default(): 
        fd={lidar_o:lidar}
        top,top_image=sess.run([tops,top_images],fd)
        np.save('/home/hhs/4T/datasets/dummy_datas/seg/lidar/lidar_%05d.npy'%i,lidar)
        np.save('/home/hhs/4T/datasets/dummy_datas/seg/top/top_%05d.npy'%i,top)
        cv2.imwrite('/home/hhs/4T/datasets/dummy_datas/seg/top_image/top_image_%05d.png'%i,top_image)
   
   
    
    
    
#
# # test
# test = np.load(bird + "000008.npy")

# print(test.shape)
# plt.imshow(test[:,:,8])
# plt.show()



