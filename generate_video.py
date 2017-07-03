import matplotlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import os
import pdb

data_root='/home/hhs/4T/datasets/dummy_datas_005/'
files_list=glob.glob(data_root+'seg/mayavi_fig/*.png')
index=np.array([file_index.strip().split('/')[-1][10:10+5] for file_index in files_list ])
num_frames=len(files_list)

video_path=data_root+'seg/result_video.avi'
fps = 24
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter(video_path,fourcc,fps,(1600,1200))#最后一个是保存图片的尺寸

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
ax0, ax1 = axes.ravel()




# fig1 = plt.subplot(2,1,1) 
# fig2 = plt.subplot(2,1,2)
# fig=plt.figure()
for i in range(num_frames):
	# i=i+300
	# fig, axes = plt.subplots(2, 1, figsize=(16, 12))
	# ax0, ax1 = axes.ravel()
	# mFig = mpimg.imread(data_root+'seg/mayavi_fig/mayavi_%05d.png'%i)
	# rgbFig=mpimg.imread(data_root+'seg/result_rgb/rgb_%05d.png'%i)
	# ax0.imshow(mFig)
	# ax0.axis('off')
	# ax1.imshow(rgbFig)
	# ax1.axis('off')
	# # fig1.imshow(mFig)
	# # fig2.imshow(rgbFig)
	# plt.tight_layout()
	# plt.axis('off')
	# plt.savefig(data_root+'seg/result_video_img/result_video_img_%05d.png'%i)
	# plt.close()
	# fig.show()
	img=cv2.imread(data_root+'seg/result_video_img/result_video_img_%05d.png'%i)
	videoWriter.write(img)
	# fig.show()
	# videoWriter.write(np.array(plt.gcf()))
	# plt.pause(0.00005)
	# plt.close()
	# pdb.set_trace()
	# plt.pause(1)
	# cv2.waitKey(1)	
videoWriter.release()