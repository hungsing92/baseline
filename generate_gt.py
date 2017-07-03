from net.common import *
from net.processing.boxes3d import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from time import time
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================



root_dir = "/home/hhs/4T/datasets/KITTI/object/training"
label_path = os.path.join(root_dir, "label_2/")
calib_path = os.path.join(root_dir, "calib/")
gt_path='/home/hhs/4T/datasets/dummy_datas/seg/gt_boxes3d'
classes = {'__background__':0, 'Car':1, 'Van':1}#, 'Van':1, 'Truck':1, 'Tram':1}

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

def project_cam2velo(cam,Tr):
	T=np.zeros([4,4],dtype=np.float32)
	T[:3,:]=Tr
	T[3,3]=1
	T_inv=np.linalg.inv(T)
	lidar_loc=np.dot(T_inv,cam)
	lidar_loc=lidar_loc[:3]
	return lidar_loc.reshape(1,3)



z=0
h_=[]
for i in range(7481):

	calib=load_kitti_calib(calib_path,i)
	Tr = calib['Tr_velo2cam']
	filename = os.path.join(label_path, str(i).zfill(6) + ".txt")
	# print("Processing: ", filename)
	with open(filename, 'r') as f:
		lines = f.readlines()

	num_objs = len(lines)
	if num_objs == 0:
		continue
	gt_boxes3d = []
	gt_labels  = []
	cam=np.ones([4,1])
	z_0=0
	z_1=0

	for j in range(num_objs):
		obj=lines[j].strip().split(' ')
		try:
			clss=classes[obj[0].strip()]
		except:
			continue
		
		h = float(obj[8])
		w = float(obj[9])
		l = float(obj[10])
		tx = float(obj[11])
		ty = float(obj[12])
		tz = float(obj[13])
		ry = float(obj[14])
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
		# pdb.set_trace()
		top_box=box3d_to_top_box([box3d])
		if (top_box[0][0]>=Top_X0) and (top_box[0][1]>=Top_Y0) and (top_box[0][2]<=Top_Xn) and (top_box[0][3]<=Top_Yn):
			# pdb.set_trace()
			gt_boxes3d.append(box3d)
			gt_labels.append(clss)
		# z_1=z_1+np.sum(box3d[:,2])/8
	# 	z_1=z_1+h
	# 	h_.append(h)
	# z=z+z_1/num_objs
	# print(z_1/num_objs)
	if len(gt_labels) == 0:
		continue
	gt_boxes3d = np.array(gt_boxes3d,dtype=np.float32)
	gt_labels  = np.array(gt_labels ,dtype=np.uint8)

	np.save('/home/hhs/4T/datasets/dummy_datas/seg/gt_boxes3d/gt_boxes3d_%05d.npy'%i,gt_boxes3d)
	np.save('/home/hhs/4T/datasets/dummy_datas/seg/gt_labels/gt_labels_%05d.npy'%i,gt_labels)
# print(z/7481)
# print(np.mean(h_))
# print(min(h_))
# print(max(h_))







