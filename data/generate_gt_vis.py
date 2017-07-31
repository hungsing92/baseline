import _init_paths
from net.common import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
import time
import glob
import cv2
from net.utility.draw import *
import mayavi.mlab as mlab
from data import *
from net.utility.file import *
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def project_velo2rgb(velo,Tr,R0,P2):
    T=np.zeros([4,4],dtype=np.float32)
    T[:3,:]=Tr
    T[3,3]=1
    R=np.zeros([4,4],dtype=np.float32)
    R[:3,:3]=R0
    R[3,3]=1
    num=len(velo)
    projections = np.zeros((num,8,2),  dtype=np.int32)
    for i in range(len(velo)):
        box3d=np.ones([8,4],dtype=np.float32)
        box3d[:,:3]=velo[i]
        M=np.dot(P2,R)
        M=np.dot(M,T)
        box2d=np.dot(M,box3d.T)
        box2d=box2d[:2,:].T/box2d[2,:].reshape(8,1)
        projections[i] = box2d
    return projections

def load_kitti_calib(calib_path,index):
    """
    load projection matrix
    """
    calib_dir = os.path.join(calib_path, str(index).zfill(6) + '.txt')

    P0 = np.zeros(12, dtype=np.float32)
    P1 = np.zeros(12, dtype=np.float32)
    P2 = np.zeros(12, dtype=np.float32)
    P3 = np.zeros(12, dtype=np.float32)
    R0 = np.zeros(9, dtype=np.float32)
    Tr_velo_to_cam = np.zeros(12, dtype=np.float32)
    Tr_imu_to_velo = np.zeros(12, dtype=np.float32)
    with open(calib_dir) as fi:
        lines = fi.readlines()
        assert(len(lines) == 8)
    
    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo= np.array(obj, dtype=np.float32)
        
    return {'P2' : P2.reshape(3,4),
            'R0' : R0.reshape(3,3),
            'Tr_velo2cam' : Tr_velo_to_cam.reshape(3, 4)}


def lidarToFront(lidar,gt_boxes3d):

    SURROUND_U_STEP = 0.3  #resolution
    SURROUND_V_STEP = 0.5
    SURROUND_U_MIN, SURROUND_U_MAX = np.array([0,    180])  # horizontal of cylindrial projection
    SURROUND_V_MIN, SURROUND_V_MAX = np.array([-45,   45])  # vertical   of cylindrial projection
    Front_V0, Front_Vn = 0, int((SURROUND_V_MAX-SURROUND_V_MIN)//SURROUND_V_STEP)+1
    Front_U0, Front_Un = 0, int((SURROUND_U_MAX-SURROUND_U_MIN)//SURROUND_U_STEP)+1

    width  = Front_Un - Front_U0+1
    height   = Front_Vn - Front_V0+1

    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]

    filter_x=np.where((pxs>=TOP_X_MIN))[0]
    filter_y=np.where((pys>=TOP_Y_MIN) & (pys<=TOP_Y_MAX))[0]
    filter_z=np.where((pzs>=TOP_Z_MIN) & (pzs<=TOP_Z_MAX))[0]
    filter_xy=np.intersect1d(filter_x,filter_y)
    filter_xyz=np.intersect1d(filter_xy,filter_z)
    pxs=pxs[filter_xyz]
    pys=pys[filter_xyz]
    pzs=pzs[filter_xyz]
    prs=prs[filter_xyz]
    velo=np.vstack([pxs,pys,pzs,np.ones_like(pxs)])
    # pdb.set_trace()

    cam_x,cam_y,cam_z=project_velo2cam(velo)
    cam_x = cam_x[:,np.newaxis]
    cam_y = cam_y[:,np.newaxis]
    cam_z = cam_z[:,np.newaxis]
    dist= np.sqrt(np.sum(cam_x**2+cam_y**2+cam_z**2,axis=1))
    angle_u=np.arctan2(cam_z,cam_x)/np.pi*180
    angle_v=np.arcsin(-cam_y/dist.reshape(-1,1))/np.pi*180
    filter_u=np.where((angle_u>=SURROUND_U_MIN) & (angle_u<=SURROUND_U_MAX))[0]
    filter_v=np.where((angle_v>=SURROUND_V_MIN) & (angle_v<=SURROUND_V_MAX))[0]
    filter_uv=np.intersect1d(filter_u,filter_v)
    dist = dist[filter_uv]
    angle_u = angle_u[filter_uv]
    angle_v = angle_v[filter_uv]
    prs=prs[filter_uv]
    # pdb.set_trace()
    qus=Front_Un-((angle_u-SURROUND_U_MIN)//SURROUND_U_STEP).astype(np.int32)
    qvs=Front_Vn-((angle_v-SURROUND_V_MIN)//SURROUND_V_STEP).astype(np.int32)
    dist_map=np.ones((height,width), dtype=np.float32)*10000
    reflect_map = np.zeros((height,width), dtype=np.float32)


    mask_map= np.zeros((height,width), dtype=np.float32)
    for i in range(len(angle_u)):
        if dist[i]<dist_map[qvs[i],qus[i]]:
            dist_map[qvs[i],qus[i]]= dist[i]
            reflect_map[qvs[i],qus[i]]= prs[i]
            mask_map[qvs[i],qus[i]]= 1
    # pdb.set_trace()       
    # dist_map[dist_map==10000]=0
    # dist_map[dist_map==0]=np.max(dist_map)
    reflect_map = reflect_map-np.min(reflect_map)
    reflect_map = (reflect_map/np.max(reflect_map)*255).astype(np.uint8) 
    reflect_map = np.dstack((reflect_map, reflect_map, reflect_map)).astype(np.uint8) 
    # cv2.imwrite('_.png',reflect_map)
    # reflect_map=cv2.imread('_.png')
    # imshow('_',reflect_map)     
    # cv2.waitKey(0)
    box3d_mask=[]
    b_box3dTo2ds=[]
    for i in range(len(gt_boxes3d)):
        box3d=gt_boxes3d[i]
        visiblity = np.zeros((8,1),dtype=np.int32)
        velo=np.hstack([box3d,np.ones([8,1])]).transpose()
        # pdb.set_trace()
        b_x,b_y,b_z=project_velo2cam(velo)
        b_x = b_x[:,np.newaxis]
        b_y = b_y[:,np.newaxis]
        b_z = b_z[:,np.newaxis]
        delta_y=b_y[4:]-b_y[:4]
        b_y[:4]=b_y[:4]+0.2*delta_y
        # b_y[4:]=b_y[4:]-0.2*delta_y
        b_dist= np.sqrt(np.sum(b_x**2+b_y**2+b_z**2,axis=1))
        b_u=np.arctan2(b_z,b_x)/np.pi*180
        b_v=np.arcsin(-b_y/b_dist.reshape(-1,1))/np.pi*180
        b_qus=Front_Un-((b_u-SURROUND_U_MIN)//SURROUND_U_STEP).astype(np.int32)
        b_qvs=Front_Vn-((b_v-SURROUND_V_MIN)//SURROUND_V_STEP).astype(np.int32)
        img_dist=dist_map[b_qvs,b_qus]
        box3d_mask.append(b_dist.reshape(-1,1)<=img_dist)
        # pdb.set_trace()
        b_box3dTo2ds.append(np.hstack([b_qus,b_qvs]))
    # pdb.set_trace()
    reflect_map = draw_rgb_projections(reflect_map, b_box3dTo2ds, color=(0,255,255), thickness=1)
    imshow('draw_box3dTo2d',reflect_map)
    pdb.set_trace()
    # cv2.waitKey(0)
    return dist_map,reflect_map,mask_map



v64 = np.arange(-24.9,2,0.43)
v16_L=np.arange(-15,2,2)-0.2
v16_H=np.arange(-15,2,2)+0.2
v64_intersect_v16=[]
for i in range(len(v64)):
    for j in range(len(v16_H)):
        if (v64[i]>v16_L[j]) and  (v64[i]<v16_H[j]):
            v64_intersect_v16.append(v64[i])
v64_intersect_v16=np.array(v64_intersect_v16)

velodyne = os.path.join(kitti_dir, "velodyne/")
img_path = os.path.join(kitti_dir, "image_2/")

label_path = os.path.join(kitti_dir, "label_2/")
calib_path = os.path.join(kitti_dir, "calib/")
# train_data_root='/home/hhs/4T/datasets/dummy_datas/seg'
classes = {'__background__':0, 'Car':1, ' Van':1}

lidar_dir = train_data_root+'/lidar16'
top_dir = train_data_root+'/top_70_16'
density_image_dir = train_data_root+'/density_image_70_16'


# pdb.set_trace()

for i in range(7481):
    # i=i+7253
    calib=load_kitti_calib(calib_path,i)
    Tr = calib['Tr_velo2cam']
    P2 = calib['P2']
    R0 = calib['R0']

    start_time=time()
    veloname = velodyne + '/'+str(i).zfill(6)+'.bin'
    print("Processing: ", veloname)
    lidar = np.fromfile(veloname, dtype=np.float32)
    lidar = lidar.reshape((-1, 4))

    filename = os.path.join(label_path, str(i).zfill(6) + ".txt")
    print("Processing: ", filename)
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_objs = len(lines)
    if num_objs == 0:
        continue
    gt_boxes3d = []
    gt_boxes2d = []
    gt_labels  = []
    cam=np.ones([4,1])

    for j in range(num_objs):
        obj=lines[j].strip().split(' ')
        try:
            clss=classes[obj[0].strip()]
            # file=open(result_path+'%06d'%i+'.txt', 'a')
            
        except:
            continue
        
        truncated = float(obj[1])
        occluded = float(obj[2])
        x1 = float(obj[4])
        y1 = float(obj[5])
        x2 = float(obj[6])
        y2 = float(obj[7])
        h = float(obj[8])
        w = float(obj[9])
        l = float(obj[10])
        tx = float(obj[11])
        ty = float(obj[12])
        tz = float(obj[13])
        ry = float(obj[14])

        # width.append(w)
        # length.append(l)
        # ratio.append(w/l)

        cam[0]=tx
        cam[1]=ty
        cam[2]=tz
        t_lidar=project_cam2velo(cam,Tr)
        Box = np.array([ # in velodyne coordinates around zero point and without orientation yet\
            [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
            [ l/2, l/2,  -l/2, -l/2, l/2, l/2,  -l/2, -l/2], \
            [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])
        rotMat = np.array([\
              [np.cos(ry), +np.sin(ry), 0.0], \
              [-np.sin(ry),  np.cos(ry), 0.0], \
              [        0.0,          0.0, 1.0]])

        cornerPosInVelo = np.dot(rotMat, Box) + np.tile(t_lidar, (8,1)).T
        box3d=cornerPosInVelo.transpose()
        box2d=np.array([x1, y1, x2, y2])
        
        gt_boxes3d.append(box3d)
        gt_boxes2d.append(box2d)
        gt_labels.append(clss)

    if len(gt_boxes3d)==0:
        continue
    # pdb.set_trace()
    rgb = cv2.imread(img_path+ '/'+str(i).zfill(6)+'.png')
    gt_3dTo2D=project_velo2rgb(gt_boxes3d,Tr,R0,P2)
    img_rcnn_nms = draw_rgb_projections(rgb, gt_3dTo2D, color=(0,0,255), thickness=1)
    imshow('draw_rcnn_nms',img_rcnn_nms)
    # cv2.waitKey(0)

    
    # v16 = lidar64To16(lidar)
    # pdb.set_trace()
    dist_map,reflect_map,mask_map=lidarToFront(lidar,gt_boxes3d)

    # reflect_map = reflect_map-np.min(reflect_map)
    # reflect_map = (reflect_map/np.max(reflect_map)*255).astype(np.uint8)
    # mask_map = mask_map-np.min(mask_map)
    # mask_map = (mask_map/np.max(mask_map)*255).astype(np.uint8)
    # imshow('reflect_map',reflect_map)
    # cv2.waitKey(0)
    # pdb.set_trace() 

    speed=time()-start_time
    print('speed: %0.4fs'%speed)

    # np.save(lidar_dir+'/lidar_%05d.npy'%i,v16)
    # np.save(top_dir+'/top_70%05d.npy'%ind[i],top_new)
    # cv2.imwrite(density_image_dir+'/density_image_70%05d.png'%ind[i],density_image)
   
   
    
    
    
#
# # test
# test = np.load(bird + "000008.npy")

# print(test.shape)
# plt.imshow(test[:,:,8])
# plt.show()



