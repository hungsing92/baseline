import _init_paths
import numpy as np
import glob
import pdb

files_list=glob.glob("/home/hhs/4T/datasets/dummy_datas/seg/gt_labels/gt_labels_*.npy")
index=np.array([file_index.strip().split('/')[-1][10:10+5] for file_index in files_list ])
num_frames=len(files_list)
train_num=int(np.round(num_frames*0.7))
random_index=np.random.permutation(index)
# pdb.set_trace()
train_list=random_index[:train_num]
val_list=random_index[train_num:]
# pdb.set_trace()
np.save('/home/hhs/4T/datasets/dummy_datas/seg/train_list.npy',train_list)
np.save('/home/hhs/4T/datasets/dummy_datas/seg/val_list.npy',val_list)
np.save('/home/hhs/4T/datasets/dummy_datas/seg/train_val_list.npy',random_index)
