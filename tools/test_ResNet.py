import _init_paths
from net.common import *
from net.utility.file import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from data.data import *

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target, anchor_filter
from net.rcnn_target_op import rcnn_target

from net.rpn_nms_op     import draw_rpn_before_nms, draw_rpn_after_nms, draw_rpn_nms
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_berfore_nms, draw_rcnn_after_nms_top,draw_rcnn_nms, rcnn_nms_2d
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels

import mayavi.mlab as mlab
import time
import glob
import tensorflow as tf
slim = tf.contrib.slim

from ResNet50_vgg_double_up_c import *
from tensorflow.python import debug as tf_debug



#---------------------------------------------------------------------------------------------
#  todo:
#    -- 3d box prameterisation
#    -- batch renormalisation
#    -- multiple image training


#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017
#<todo>

def generat_test_reslut(probs, boxes3d, rgb_shape, index, boxes2d=None ):

    # empty(result_path)
    if len(boxes3d)==0:
        return 1
    file=open(result_path+'%06d'%index+'.txt', 'w')
    if boxes2d==None:
        rgb_boxes=project_to_rgb_roi(boxes3d, rgb_shape[1], rgb_shape[0] )
        rgb_boxes=rgb_boxes[:,1:5]
    else:
        rgb_boxes=boxes2d
    reuslts=[]
    # pdb.set_trace()
    for num in np.arange(len(probs)):
        box= rgb_boxes[num]
        box3d = boxes3d[num]
        center = np.sum(box3d,axis=0, keepdims=True)/8
        # pdb.set_trace()
        dis=0
        for k in [0, 2, 4, 6]:
            i,j=k,k+1
            dis +=np.sum((box3d[i]-box3d[j])**2) **0.5
        w = dis/4
        dis=0
        for k in [3, 7]:
            i,j=k,k-3
            dis +=np.sum((box3d[i]-box3d[j])**2) **0.5
            i,j=k-2,k-1
            dis +=np.sum((box3d[i]-box3d[j])**2) **0.5
        l = dis/4
        dis=0
        for k in range(0,4):
            i,j=k,k+4
            dis +=np.sum((box3d[i]-box3d[j])**2) **0.5
        h = dis/4

        x = center[:,0]
        y = center[:,1]
        z = center[:,2]-h/2
        velo=np.array([x,y,z,1]).reshape(4,1)
        tx,ty,tz = project_velo2cam(velo)

        x1 = float(box3d[3,0])
        y1 = float(box3d[3,1])
        x2 = float(box3d[0,0])
        y2 = float(box3d[0,1])
        vect = (x2-x1,y2-y1)
        ry= np.arctan((x1-x2)/-(y2-y1))
        if vect[0]>0:
            if ry<0:
                ry = ry + np.pi
        else:
            if ry>0:
                ry = ry - np.pi
        # pdb.set_trace()
        if probs == [] :
            line='Car -1 -1 -10 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%(box[0], box[1], box[2], box[3],h,w,l,tx,ty,tz,ry,1-(i+1)*0.06)
            file.write(line)
        else:
            line='Car -1 -1 -10 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n'%(box[0], box[1], box[2], box[3],h,w,l,tx,ty,tz,ry,probs[num])
            file.write(line)

    file.close()
    return 1

def load_dummy_datas(index):

    num_frames = []
    rgbs      =[]
    lidars    =[]
    tops      =[]
    fronts    =[]
    gt_labels =[]
    gt_boxes3d=[]
    rgbs_norm =[]
    top_images  =[]
    front_images=[]

    rgb   = cv2.imread(kitti_dir+'/image_2/%06d.png'%int(index),1).astype(np.float32, copy=False)
    rgbs_norm0=(rgb-PIXEL_MEANS)/255
    lidar = np.load(train_data_root+'/lidar/lidar_%05d.npy'%int(index))
    top   = np.load(train_data_root+'/top_70_0.1/top_70%05d.npy'%int(index))
    front = np.zeros((1,1),dtype=np.float32)
    gt_label  = np.load(train_data_root+'/gt_labels/gt_labels_%05d.npy'%int(index))
    gt_box3d = np.load(train_data_root+'/gt_boxes3d/gt_boxes3d_%05d.npy'%int(index))
    rgb_shape   = rgb.shape
    # gt_rgb   = project_to_rgb_roi  (gt_box3d, rgb_shape[1], rgb_shape[0] )
    # keep = np.where((gt_rgb[:,1]>=-200) & (gt_rgb[:,2]>=-200) & (gt_rgb[:,3]<=(rgb_shape[1]+200)) & (gt_rgb[:,4]<=(rgb_shape[0]+200)))[0]
    # gt_label=gt_label[keep]
    # gt_box3d=gt_box3d[keep]
    top_image   = cv2.imread(train_data_root+'/density_image_70_0.1/density_image_70%05d.png'%int(index))
    front_image = np.zeros((1,1,3),dtype=np.float32)
    rgbs.append(rgb)
    lidars.append(lidar)
    tops.append(top)
    fronts.append(front)
    gt_labels.append(gt_label)
    gt_boxes3d.append(gt_box3d)
    top_images.append(top_image)
    front_images.append(front_image)
    rgbs_norm.append(rgbs_norm0)
    # explore dataset:
    # print (gt_box3d)
    if 0:
        fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 1000))
        projections=box3d_to_rgb_projections(gt_box3d)
        rgb1 = draw_rgb_projections(rgb, projections, color=(255,255,255), thickness=2)
        top_image1 = draw_box3d_on_top(top_image, gt_box3d, color=(255,255,255), thickness=2)
        rgb_boxes=project_to_rgb_roi(gt_box3d, rgb_shape[1], rgb_shape[0] )
                # rgb_boxes=batch_rgb_rois
        img_rgb_2d_detection = draw_boxes(rgb, rgb_boxes[:,1:5], color=(255,0,255), thickness=1)
        imshow('img_rgb_2d_detection',img_rgb_2d_detection)
        imshow('rgb',rgb1)
        imshow('top_image',top_image1)
        mlab.clf(fig)
        draw_lidar(lidar, fig=fig)
        draw_gt_boxes3d(gt_box3d, fig=fig)
        mlab.show(1)
        cv2.waitKey(0)
        pass
    # pdb.set_trace()
    # rgbs=np.array(rgbs)
    ##exit(0)
    mlab.close(all=True)
    return  rgbs, tops, fronts, gt_labels, gt_boxes3d, top_images, front_images, lidars, rgbs_norm


is_show=0
# MM_PER_VIEW1 = 120, 30, 70, [1,1,0]
MM_PER_VIEW1 = 180, 70, 60, [1,1,0]#[ 12.0909996 , -1.04700089, -2.03249991]
result_path='./evaluate_object/val_R_v/data/'
result_path_plot='./evaluate_object/val_R_v/plot/'
empty(result_path)
empty(result_path_plot)
makedirs(result_path)
makedirs(result_path_plot)

def run_test():

    # output dir, etc
    out_dir = './outputs'
    makedirs(out_dir +'/tf')
    makedirs(out_dir +'/check_points')
    log = Logger(out_dir+'/log_%s.txt'%(time.strftime('%Y-%m-%d %H:%M:%S')),mode='a')

    # index=np.load(train_data_root+'/val_list.npy')
    index_file=open(train_data_root+'/val.txt')
    # index_file=open(train_data_root+'/train.txt')
    index = [ int(i.strip()) for i in index_file]
    index_file.close()
    
    index=sorted(index)
    print('len(index):%d'%len(index))
    num_frames=len(index)
    #lidar data -----------------
    if 1:
        ratios_rgb=np.array([0.3,0.6,.75,1], dtype=np.float32)
        scales_rgb=np.array([0.5,1,2,4],   dtype=np.float32)
        bases_rgb = make_bases(
            base_size = 48,
            ratios=ratios_rgb,
            scales=scales_rgb
        )
        ratios=np.array([1.7,2.4])
        scales=np.array([1.7,2.4])
        bases=np.array([
            [-12,-6,12,6],
            [-6,-12,6,12],
            [-17,-8,17,8],
            [-8,-17,8,17],
            [-22,-8,22,8],
            [-8,-22,8,22],
            [-17,-9,17,9],
            [-9,-17,9,17],
            [-22,-9,22,9],
            [-9,-22,9,22],
            [-27,-10,27,10],
            [-10,-27,10,27],
                        ])
        num_bases = len(bases)
        num_bases_rgb = len(bases_rgb)
        stride = 4

        rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, lidars,rgbs_norm0 = load_dummy_datas(index[10])
        # num_frames = len(rgbs)

        top_shape   = tops[0].shape
        front_shape = fronts[0].shape
        rgb_shape   = rgbs[0].shape
        top_feature_shape = ((top_shape[0]-1)//stride+1, (top_shape[1]-1)//stride+1)
        rgb_feature_shape = ((rgb_shape[0]-1)//stride+1, (rgb_shape[1]-1)//stride+1)
        out_shape=(8,3)


        #-----------------------
        #check data
        if 0:
            fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 1000))
            draw_lidar(lidars[0], fig=fig)
            draw_gt_boxes3d(gt_boxes3d[0], fig=fig)
            mlab.show(1)
            cv2.waitKey(0)



    # set anchor boxes
    num_class = 2 #incude background
    anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
    anchors_rgb, inside_inds_rgb =  make_anchors(bases_rgb, stride, rgb_shape[0:2], rgb_feature_shape[0:2])
    print ('out_shape=%s'%str(out_shape))
    print ('num_frames=%d'%num_frames)


    #load model ####################################################################################################
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')
    rgb_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors_rgb'    )
    rgb_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds_rgb')

    top_images   = tf.placeholder(shape=[None, *top_shape  ], dtype=tf.float32, name='top'  )
    front_images = tf.placeholder(shape=[None, *front_shape], dtype=tf.float32, name='front')
    rgb_images   = tf.placeholder(shape=[None, None, None, 3 ], dtype=tf.float32, name='rgb'  )
    top_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'   ) #<todo> change to int32???
    front_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='front_rois' )
    rgb_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'   )

    top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores,deltasZ,proposals_z,inside_inds_nms  = \
        top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)
   
    front_features = front_feature_net(front_images)
    rgb_features, rgb_scores, rgb_probs, rgb_deltas  = rgb_feature_net(rgb_images, num_bases_rgb)

    fuse_scores, fuse_probs, fuse_deltas, fuse_deltas_2d = \
        fusion_net(
            ( [top_features,     top_rois,     7,7,1./stride],
              [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
              [rgb_features,     rgb_rois,     7,7,1./(1*stride)],
              # [top_features,     top_rois,     7,7,1./(0.75*stride)],
              # [front_features,   front_rois,   0,0,1./(0.75*stride)],  #disable by 0,0
              # [rgb_features,     rgb_rois,     7,7,1./(0.75*stride)],
              ),
            num_class, out_shape) #<todo>  add non max suppression


    num_ratios=len(ratios)
    num_scales=len(scales)
    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver()  
        saver.restore(sess, './outputs/check_points/snap_R2R_new_fusesion_augment_pos_samples_01_065000.ckpt')

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0

        for iter in range(num_frames):
            start_time=time.time()
            # iter=iter+20
            print('Processing Img: %d  %s'%(iter, index[iter]))
            rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, lidars,rgbs_norm0 = load_dummy_datas(index[iter])
            idx=0
            rgb_shape   = rgbs[idx].shape
            # top_img=top_imgs[idx]

            batch_top_images    = tops[idx].reshape(1,*top_shape)
            batch_front_images  = fronts[idx].reshape(1,*front_shape)
            batch_rgb_images    = rgbs_norm0[idx].reshape(1,*rgb_shape)


            batch_gt_labels    = gt_labels[idx]
            batch_gt_boxes3d   = gt_boxes3d[idx]
            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d)

            inside_inds_filtered=anchor_filter(batch_top_images[0,:,:,-1], anchors, inside_inds)
            inside_inds_filtered_rgb=inside_inds_rgb

            ## run propsal generation ------------
            fd1={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds_filtered,

                rgb_images:      batch_rgb_images,
                rgb_anchors:     anchors_rgb,
                rgb_inside_inds: inside_inds_filtered_rgb,

                IS_TRAIN_PHASE:  False
            }
            batch_proposals, batch_proposal_scores, batch_top_features, batch_top_proposals_z = sess.run([proposals, proposal_scores, top_features,proposals_z],fd1)
            # pdb.set_trace()
            print('nums of proposals:%d'%len(batch_proposals))
            batch_top_rois = batch_proposals
            batch_rois3d = top_z_to_box3d(batch_top_rois[:,1:5],batch_top_proposals_z)
            
            batch_rois3d_old  = project_to_roi3d(batch_top_rois)
            batch_front_rois = project_to_front_roi(batch_rois3d )
            batch_rgb_rois      = project_to_rgb_roi     (batch_rois3d , rgb_shape[1], rgb_shape[0] )
            batch_rgb_rois_old      = project_to_rgb_roi     (batch_rois3d_old , rgb_shape[1], rgb_shape[0] )

            ## run classification and regression  -----------

            fd2={
                **fd1,

                top_images:      batch_top_images,
                front_images:    batch_front_images,
                rgb_images:      batch_rgb_images,

                top_rois:        batch_top_rois,
                front_rois:      batch_front_rois,
                rgb_rois:        batch_rgb_rois,

            }
            # batch_top_probs,  batch_top_deltas  =  sess.run([ top_probs,  top_deltas  ],fd2)
            batch_fuse_probs, batch_fuse_deltas, batch_fuse_deltas_2d =  sess.run([ fuse_probs, fuse_deltas, fuse_deltas_2d ],fd2)
            probs, boxes3d, boxes2d = rcnn_nms_2d(batch_fuse_probs, batch_fuse_deltas, batch_rois3d_old, batch_fuse_deltas_2d, batch_rgb_rois_old[:,1:], rgb_shape, threshold=0.05)
            # print('nums of boxes3d : %d'%len(boxes3d))
            generat_test_reslut(probs, boxes3d, rgb_shape, int(index[iter]), boxes2d)
            speed=time.time()-start_time
            print('speed: %0.4fs'%speed)
            # pdb.set_trace()
            # debug: ------------------------------------
            if is_show == 1:
                top_image      = top_imgs[idx]
                surround_image = fronts[idx]
                lidar = lidars[idx]
                rgb=rgbs[idx]

                batch_top_probs, batch_top_scores, batch_top_deltas  = \
                    sess.run([ top_probs, top_scores, top_deltas ],fd2)
                batch_fuse_probs, batch_fuse_deltas = \
                    sess.run([ fuse_probs, fuse_deltas ],fd2)    
                ## show on lidar
                mfig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 1000))
 
                draw_lidar(lidar, fig=mfig)
                # if len(boxes3d)!=0:
                    # draw_didi_boxes3d(mfig, boxes3d)
                draw_target_boxes3d(boxes3d, fig=mfig)
                draw_gt_boxes3d(batch_gt_boxes3d, fig=mfig)
                azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
                mlab.view(azimuth,elevation,distance,focalpoint)
                mlab.show(1)
                # # cv2.waitKey(0)
                # # mlab.close()               
                img_rpn_nms = draw_rpn_nms(top_image, batch_proposals, batch_proposal_scores)
                img_gt     = draw_rpn_gt(img_rpn_nms, batch_gt_top_boxes, batch_gt_labels)
                # imshow('img_rpn_nms',img_gt)
                cv2.waitKey(1)
                # imshow('img_rpn_gt',img_gt)

                rgb1 =draw_rcnn_nms (rgb, boxes3d, probs)                
                projections=box3d_to_rgb_projections(batch_gt_boxes3d)
                img_rcnn_nms = draw_rgb_projections(rgb1, projections, color=(0,0,255), thickness=1)

                # pdb.set_trace()
                rgb_boxes=project_to_rgb_roi(boxes3d, rgb_shape[1], rgb_shape[0] )
                # rgb_boxes=batch_rgb_rois
                img_rgb_2d_detection = draw_boxes(rgb, boxes2d, color=(255,0,255), thickness=1)
                img_rgb_3d_2_2d = draw_boxes(rgb, rgb_boxes[:,1:], color=(255,0,255), thickness=1)

                imshow('draw_rcnn_nms',img_rcnn_nms)
                imshow('img_rgb_2d_detection',img_rgb_2d_detection)
                # imshow('img_rgb_3d_2_2d',img_rgb_3d_2_2d)
                cv2.waitKey(500)
                # plt.pause(0.25)
                # mlab.clf(mfig)

## main function ##########################################################################


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ##run_train()
    run_test()
