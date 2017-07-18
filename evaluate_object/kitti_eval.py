import numpy as np
eval_list = []
res_file_car_precision = "./mv3d_lidar_kitti_val/data/stats_car_detection.txt"
res_file_car_recall = "./mv3d_lidar_kitti_val/data/stats_car_recall.txt"
with open(res_file_car_precision) as f:
        for mode in ['easy', 'medium', 'hard']:
            line = f.readline()
            result = np.array(line.rstrip().split(" ")).astype(float)
            mean = np.mean(result)
            print "The mAP of car:%f" % mean
            eval_list.append(("val   " + mode, mean))
with open(res_file_car_recall) as f:
        for mode in ['easy', 'medium', 'hard']:
            line = f.readline()
            result = np.array(line.rstrip().split(" ")).astype(float)
            mean = np.mean(result)
            print "The recall of car:%f" % mean
            eval_list.append(("val   " + mode, mean))
