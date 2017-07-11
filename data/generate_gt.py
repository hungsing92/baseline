import _init_paths
from net.common import *
from net.processing.boxes3d import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
import tensorflow as tf
from time import time
from net.utility.file import *
# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================

def project_velo2cam(velo,Tr,R0,P2):
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

def project_to_rgb_roi(rois3d, width, height,Tr,R0,P2):
    num  = len(rois3d)
    rois = np.zeros((num,5),dtype=np.int32)
    projections = project_velo2cam(rois3d,Tr,R0,P2)
    for n in range(num):
        qs = projections[n]
        minx = np.min(qs[:,0])
        maxx = np.max(qs[:,0])
        miny = np.min(qs[:,1])
        maxy = np.max(qs[:,1])
        minx = np.maximum(np.minimum(minx, width - 1), 0)
        maxx = np.maximum(np.minimum(maxx, width - 1), 0)
        miny = np.maximum(np.minimum(miny, height - 1), 0)
        maxy = np.maximum(np.minimum(maxy, height - 1), 0)
        rois[n,1:5] = minx,miny,maxx,maxy

    return rois

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
	lidar_loc_=np.dot(T_inv,cam)
	lidar_loc=lidar_loc_[:3]
	return lidar_loc.reshape(1,3)

# kitti_dir = "/home/hhs/4T/datasets/KITTI/object/training"
label_path = os.path.join(kitti_dir, "label_2/")
calib_path = os.path.join(kitti_dir, "calib/")
# train_data_root='/home/hhs/4T/datasets/dummy_datas/seg'
classes = {'__background__':0, 'Car':1}#, ' Van':1, 'Truck':1, 'Tram':1}
# result_path='./evaluate_object/val_gt/'
gt_boxes3d_path = train_data_root + '/gt_boxes3d'
gt_labels_path = train_data_root + '/gt_labels'

empty(gt_boxes3d_path)
empty(gt_labels_path)

for i in range(7481):

	calib=load_kitti_calib(calib_path,i)
	Tr = calib['Tr_velo2cam']
	P2 = calib['P2']
	R0 = calib['R0']
	filename = os.path.join(label_path, str(i).zfill(6) + ".txt")
	print("Processing: ", filename)
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
			# file=open(result_path+'%06d'%i+'.txt', 'a')
			
		except:
			continue
		
		truncated = float(obj[1])
		occluded = float(obj[2])
		h = float(obj[8])
		w = float(obj[9])
		l = float(obj[10])
		tx = float(obj[11])
		ty = float(obj[12])
		tz = float(obj[13])
		ry = float(obj[14])
		if 	tx==-1000:
			print(lines[j].strip())
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

		# rgb_boxes=project_to_rgb_roi([box3d], 1242, 375 ,Tr,R0,P2)
		# line='Car %.2f %d -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.2f\n'%(truncated, occluded, rgb_boxes[0][1], rgb_boxes[0][2], rgb_boxes[0][3], rgb_boxes[0][4], 1)
		# file.write(line)

		top_box=box3d_to_top_box([box3d])
		if (top_box[0][0]>=Top_X0) and (top_box[0][1]>=Top_Y0) and (top_box[0][2]<=Top_Xn) and (top_box[0][3]<=Top_Yn):

			gt_boxes3d.append(box3d)
			gt_labels.append(clss)

	if len(gt_labels) == 0:
		continue

	gt_boxes3d = np.array(gt_boxes3d,dtype=np.float32)
	gt_labels  = np.array(gt_labels ,dtype=np.uint8)
	file.close()

	np.save(gt_boxes3d_path+'/gt_boxes3d_%05d.npy'%i,gt_boxes3d)
	np.save(gt_labels_path+'/gt_labels_%05d.npy'%i,gt_labels)




#Generate train and val list
#3DOP train val list http://www.cs.toronto.edu/objprop3d/data/ImageSets.tar.gz
files_list=glob.glob(gt_labels_path+"/gt_labels_*.npy")
index=np.array([file_index.strip().split('_')[-1].split('.')[0] for file_index in files_list ])
num_frames=len(files_list)
train_num=int(np.round(num_frames*0.7))
random_index=np.random.permutation(index)
train_list=random_index[:train_num]
val_list=random_index[train_num:]
np.save(train_data_root+'/train_list.npy',train_list)
np.save(train_data_root+'/val_list.npy',val_list)
np.save(train_data_root+'/train_val_list.npy',random_index)






