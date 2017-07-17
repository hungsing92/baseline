import _init_paths
from net.common import *
from net.utility.file import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

# from dummynet import *
from data.data import *

from net.rpn_loss_op import *
from net.rcnn_loss_op import *
from net.rpn_target_op import make_bases, make_anchors, rpn_target, anchor_filter
from net.rcnn_target_op import rcnn_target, rcnn_target_ohem

from net.rpn_nms_op     import draw_rpn_nms, draw_rpn
from net.rcnn_nms_op    import rcnn_nms, draw_rcnn_nms, draw_rcnn
from net.rpn_target_op  import draw_rpn_gt, draw_rpn_targets, draw_rpn_labels
from net.rcnn_target_op import draw_rcnn_targets, draw_rcnn_labels

import mayavi.mlab as mlab

import time
import glob
import tensorflow as tf
slim = tf.contrib.slim
# from mobilenet import *
from ResNet50_vgg_double_up_c import *
from tensorflow.python import debug as tf_debug
# os.environ["QT_API"] = "pyqt"

#http://3dimage.ee.tsinghua.edu.cn/cxz
# "Multi-View 3D Object Detection Network for Autonomous Driving" - Xiaozhi Chen, CVPR 2017
MM_PER_VIEW1 = 180, 70, 30, [1,1,0]
def load_dummy_datas(index):

    num_frames = []
    rgbs      =[]
    # lidars    =[]
    tops      =[]
    fronts    =[]
    gt_labels =[]
    gt_boxes3d=[]
    rgbs_norm =[]
    top_images  =[]
    front_images=[]

    # pdb.set_trace()
    if num_frames==[]:
        num_frames=len(index)
        print('num_frames:%d'%num_frames)
    for n in range(num_frames):
        print('processing img:%d,%05d'%(n,int(index[n])))
        rgb   = cv2.imread(kitti_dir+'/image_2/%06d.png'%int(index[n]))
        rgbs_norm0=(rgb-PIXEL_MEANS)/255
        # lidar = np.load(train_data_root+'/lidar/lidar_%05d.npy'%int(index[n]))
        top   = np.load(train_data_root+'/top_70/top_70%05d.npy'%int(index[n]))
        front = np.zeros((1,1),dtype=np.float32)
        gt_label  = np.load(train_data_root+'/gt_labels/gt_labels_%05d.npy'%int(index[n]))
        gt_box3d = np.load(train_data_root+'/gt_boxes3d/gt_boxes3d_%05d.npy'%int(index[n]))
        top_image   = cv2.imread(train_data_root+'/density_image_70/density_image_70%05d.png'%int(index[n]))
        front_image = np.zeros((1,1,3),dtype=np.float32)

        rgbs.append(rgb)
        # lidars.append(lidar)
        tops.append(top)
        fronts.append(front)
        gt_labels.append(gt_label)
        gt_boxes3d.append(gt_box3d)
        top_images.append(top_image)
        front_images.append(front_image)
        rgbs_norm.append(rgbs_norm0)

        # explore dataset:
        if 0:
            fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            projections=box3d_to_rgb_projections(gt_box3d)
            rgb1 = draw_rgb_projections(rgb, projections, color=(255,255,255), thickness=2)
            top_image1 = draw_box3d_on_top(top_image, gt_box3d, color=(255,255,255), thickness=2)

            imshow('rgb',rgb1)
            imshow('top_image',top_image1)
            azimuth,elevation,distance,focalpoint = MM_PER_VIEW1
            mlab.view(azimuth,elevation,distance,focalpoint)
            mlab.clf(fig)
            draw_lidar(lidar, fig=fig)
            draw_gt_boxes3d(gt_box3d, fig=fig)
            mlab.show(1)
            cv2.waitKey(0)
            mlab.close()
            pass
    # pdb.set_trace()
    # rgbs=np.array(rgbs)
    ##exit(0)
    mlab.close(all=True)
    return  rgbs, tops, fronts, gt_labels, gt_boxes3d, top_images, front_images, rgbs_norm, index#, lidars


vis=0
ohem=True
def run_train():

    # output dir, etc
    out_dir = './outputs'
    makedirs(out_dir +'/tf')
    makedirs(out_dir +'/check_points')
    makedirs(out_dir +'/log')
    log = Logger(out_dir+'/log/log_%s.txt'%(time.strftime('%Y-%m-%d %H:%M:%S')),mode='a')
    index_file=open(train_data_root+'/train.txt')
    index=[ int(i.strip()) for i in index_file]
    index=sorted(index)
    index=np.array(index)
    num_frames = len(index)
    # pdb.set_trace()
    #lidar data -----------------
    if 1:
        ###generate anchor base 
        # ratios=np.array([0.4,0.6,1.7,2.4], dtype=np.float32)
        # scales=np.array([0.5,1,2,3],   dtype=np.float32)
        # bases = make_bases(
        #     base_size = 16,
        #     ratios=ratios,
        #     scales=scales
        # )
        ratios=np.array([1.7,2.4,2.7])
        scales=np.array([1.7,2.4])
        bases=np.array([[-19.5, -8, 19.5, 8],
                        [-8, -19.5, 8, 19.5],
                        [-27.5, -11, 27.5, 11],
                        [-11, -27.5, 11, 27.5],
                        [-5, -3, 5, 3],
                        [-3, -5, 3, 5]
                        ])
        # pdb.set_trace()
        num_bases = len(bases)
        stride = 4

        out_shape=(8,3)

        rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, rgbs_norm, image_index = load_dummy_datas(index[:3])
        # rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, rgbs_norm, image_index, lidars = load_dummy_datas()
        top_shape   = tops[0].shape
        front_shape = fronts[0].shape
        rgb_shape   = rgbs[0].shape
        top_feature_shape = ((top_shape[0]-1)//stride+1, (top_shape[1]-1)//stride+1)
        # set anchor boxes
        num_class = 2 #incude background
        anchors, inside_inds =  make_anchors(bases, stride, top_shape[0:2], top_feature_shape[0:2])
        # inside_inds = np.arange(0,len(anchors),dtype=np.int32)  #use all  #<todo>
        print ('out_shape=%s'%str(out_shape))
        print ('num_frames=%d'%num_frames)

        #-----------------------
        #check data
        if 0:
            fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1000, 500))
            draw_lidar(lidars[0], fig=fig)
            draw_gt_boxes3d(gt_boxes3d[0], fig=fig)
            mlab.show(1)
            cv2.waitKey(1)

    #load model ####################################################################################################
    top_anchors     = tf.placeholder(shape=[None, 4], dtype=tf.int32,   name ='anchors'    )
    top_inside_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name ='inside_inds')

    top_images   = tf.placeholder(shape=[None, *top_shape  ], dtype=tf.float32, name='top'  )
    front_images = tf.placeholder(shape=[None, *front_shape], dtype=tf.float32, name='front')
    rgb_images   = tf.placeholder(shape=[None, None, None, 3 ], dtype=tf.float32, name='rgb'  )
    top_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='top_rois'   ) #<todo> change to int32???
    front_rois   = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='front_rois' )
    rgb_rois     = tf.placeholder(shape=[None, 5], dtype=tf.float32,   name ='rgb_rois'   )

    top_features, top_scores, top_probs, top_deltas, proposals, proposal_scores = \
        top_feature_net(top_images, top_anchors, top_inside_inds, num_bases)
    # pdb.set_trace()
    front_features = front_feature_net(front_images)
    rgb_features   = rgb_feature_net(rgb_images) 

    fuse_scores, fuse_probs, fuse_deltas = \
        fusion_net(
			( [top_features,     top_rois,     7,7,1./stride],
			  [front_features,   front_rois,   0,0,1./stride],  #disable by 0,0
			  [rgb_features,     rgb_rois,     7,7,1./(1*stride)],),
            num_class, out_shape) #<todo>  add non max suppression

    #loss ########################################################################################################
    top_inds     = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_ind'    )
    top_pos_inds = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_pos_ind')
    top_labels   = tf.placeholder(shape=[None   ], dtype=tf.int32,   name='top_label'  )
    top_targets  = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target' )
    top_cls_loss, top_reg_loss = rpn_loss(2*top_scores, top_deltas, top_inds, top_pos_inds, top_labels, top_targets)

    fuse_labels  = tf.placeholder(shape=[None            ], dtype=tf.int32,   name='fuse_label' )
    fuse_targets = tf.placeholder(shape=[None, *out_shape], dtype=tf.float32, name='fuse_target')
    
    fuse_scores_ohem=tf.stop_gradient(fuse_scores)
    fuse_deltas_ohem=tf.stop_gradient(fuse_deltas)
    fuse_labels_ohem=tf.stop_gradient(fuse_labels)
    fuse_targets_ohem=tf.stop_gradient(fuse_targets)
    softmax_loss_ohem, rcnn_smooth_l1_ohem = rcnn_loss_ohem(fuse_scores_ohem, fuse_deltas_ohem, fuse_labels_ohem, fuse_targets_ohem)

    fuse_cls_loss, fuse_reg_loss = rcnn_loss(fuse_scores, fuse_deltas, fuse_labels, fuse_targets)
    tf.summary.scalar('rpn_cls_loss', top_cls_loss)
    tf.summary.scalar('rpn_reg_loss', top_reg_loss)
    tf.summary.scalar('rcnn_cls_loss', fuse_cls_loss)
    tf.summary.scalar('rcnn_reg_loss', fuse_reg_loss)

    #solver
    l2 = l2_regulariser(decay=0.00001)
    tf.summary.scalar('l2', l2)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    rate=0.0005
    solver = tf.train.AdamOptimizer(learning_rate)
    solver_step = solver.minimize(1*top_cls_loss+1*top_reg_loss+1.5*fuse_cls_loss+2*fuse_reg_loss+l2)

    max_iter = 200000
    iter_debug=1

    # start training here  #########################################################################################
    log.write('epoch     iter    speed   rate   |  top_cls_loss   reg_loss   |  fuse_cls_loss  reg_loss  |  \n')
    log.write('-------------------------------------------------------------------------------------\n')

    num_ratios=len(ratios)
    num_scales=len(scales)
    # fig, axs = plt.subplots(num_ratios,num_scales)

    merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()  
    train_writer = tf.summary.FileWriter( './outputs/tensorboard/RVD_ohem_NL',
                                      sess.graph)
    with sess.as_default():
        sess.run( tf.global_variables_initializer(), { IS_TRAIN_PHASE : True } )
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # summary_writer = tf.summary.FileWriter(out_dir+'/tf', sess.graph)
        saver  = tf.train.Saver() 

        # saver_0.restore(sess, './outputs/check_points/snap_RVD_FreezeBN_NGT_OHEM_s_070000.ckpt') 
        saver.restore(sess, './outputs/check_points/snap_RVD_new_lidar_6s_090000.ckpt') 
 

        # var_lt_res=[v for v in tf.trainable_variables() if v.name.startswith('res')]#resnet_v1_50
        # # pdb.set_trace()
        # ## var_lt=[v for v in tf.trainable_variables() if not(v.name.startswith('fuse-block-1')) and not(v.name.startswith('fuse')) and not(v.name.startswith('fuse-input'))]

        # # # var_lt.pop(0)
        # # # var_lt.pop(0)
        # # # pdb.set_trace()
        # saver_0=tf.train.Saver(var_lt_res)        
        # # # 
        # saver_0.restore(sess, './outputs/check_points/resnet_v1_50.ckpt')
        # # # pdb.set_trace()
        # # top_lt=[v for v in tf.trainable_variables() if v.name.startswith('top_base')]
        # # top_lt.pop(0)
        # # # # top_lt.pop(0)
        # # for v in top_lt:
        # #     # pdb.set_trace()
        # #     for v_rgb in var_lt:
        # #         if v.name[9:]==v_rgb.name:
        # #             print ("assign weights:%s"%v.name)
        # #             v.assign(v_rgb)
        # var_lt_vgg=[v for v in tf.trainable_variables() if v.name.startswith('vgg')]
        # var_lt_vgg.pop(0)
        # saver_1=tf.train.Saver(var_lt_vgg)
        
        # # # pdb.set_trace()
        # saver_1.restore(sess, './outputs/check_points/vgg_16.ckpt')

        batch_top_cls_loss =0
        batch_top_reg_loss =0
        batch_fuse_cls_loss=0
        batch_fuse_reg_loss=0
        # # rate=0.00005
        frame_range = np.arange(num_frames)
        idx=0
        frame=0
        rate=0.00001
        for iter in range(max_iter):
            epoch=iter//num_frames+1
            # rate=0.001
            start_time=time.time()
            # generate train image -------------
            # idx = np.random.choice(num_frames)     #*10   #num_frames)  #0
            # shuffle the samples every 4*num_frames
            if iter%(num_frames*2)==0:
                idx=0
                frame=0
                count=0
                end_flag=0
                frame_range1 = np.random.permutation(num_frames)
                if np.all(frame_range1==frame_range):
                    raise Exception("Invalid level!", permutation)
                frame_range=frame_range1

            #load freq samples every 2*freq iterations
            freq=int(200)
            if idx%freq==0 :
                count+=idx
                if count%(2*freq)==0:
                    frame+=idx
                    frame_end=min(frame+freq,num_frames)
                    if frame_end==num_frames:
                        end_flag=1
                    # pdb.set_trace()
                    del rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, rgbs_norm, image_index
                    rgbs, tops, fronts, gt_labels, gt_boxes3d, top_imgs, front_imgs, rgbs_norm, image_index = load_dummy_datas(index[frame_range[frame:frame_end]])
                idx=0
            if (end_flag==1) and (idx+frame)==num_frames:
                idx=0
            print('processing image : %s'%image_index[idx])

            # if (iter+1)%(20000)==0:
            #     rate=0.8*rate

            rgb_shape   = rgbs[idx].shape
            batch_top_images    = tops[idx].reshape(1,*top_shape)
            batch_front_images  = fronts[idx].reshape(1,*front_shape)
            batch_rgb_images    = rgbs_norm[idx].reshape(1,*rgb_shape)
            # batch_rgb_images    = rgbs[idx].reshape(1,*rgb_shape)

            top_img=tops[idx]
            inside_inds_filtered=anchor_filter(top_img[:,:,-1], anchors, inside_inds)

            batch_gt_labels    = gt_labels[idx]
            if len(batch_gt_labels)==0:
                idx=idx+1
                continue
            batch_gt_boxes3d   = gt_boxes3d[idx]
            # pdb.set_trace()
            batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d)

			## run propsal generation ------------
            fd1={
                top_images:      batch_top_images,
                top_anchors:     anchors,
                top_inside_inds: inside_inds_filtered,

                learning_rate:   rate,
                IS_TRAIN_PHASE:  True
            }

            batch_proposals, batch_proposal_scores, batch_top_features = sess.run([proposals, proposal_scores, top_features],fd1)
            ## generate  train rois  ------------
            batch_top_inds, batch_top_pos_inds, batch_top_labels, batch_top_targets  = \
                rpn_target ( anchors, inside_inds_filtered, batch_gt_labels,  batch_gt_top_boxes)

            if ohem==True:
                batch_top_rois, batch_fuse_labels, batch_fuse_targets  = \
                     rcnn_target_ohem(  batch_proposals, batch_gt_labels, batch_gt_top_boxes, batch_gt_boxes3d )
        
                batch_rois3d     = project_to_roi3d    (batch_top_rois)
                batch_front_rois = project_to_front_roi(batch_rois3d  ) 
                batch_rgb_rois   = project_to_rgb_roi  (batch_rois3d, rgb_shape[1], rgb_shape[0])
    
    
                ## run classification and regression loss -----------
                fd2={
                **fd1,
    
                    top_images: batch_top_images,
                    front_images: batch_front_images,
                    rgb_images: batch_rgb_images,
    
                    top_rois:   batch_top_rois,
                    front_rois: batch_front_rois,
                    rgb_rois:   batch_rgb_rois,
    
                    top_inds:     batch_top_inds,
                    top_pos_inds: batch_top_pos_inds,
                    top_labels:   batch_top_labels,
                    top_targets:  batch_top_targets,
    
                    fuse_labels:  batch_fuse_labels,
                    fuse_targets: batch_fuse_targets,
                }
                #_, batch_top_cls_loss, batch_top_reg_loss = sess.run([solver_step, top_cls_loss, top_reg_loss],fd2)
    
                rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
                fg_rois_per_image = int(np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image))
                loss_ohem_, rcnn_smooth_l1_ohem_= sess.run([softmax_loss_ohem, rcnn_smooth_l1_ohem],fd2)
                # loss_ohem_[:len(rcnn_smooth_l1_ohem_)] += rcnn_smooth_l1_ohem_
                fg_inds=np.arange(len(rcnn_smooth_l1_ohem_))
                if len(rcnn_smooth_l1_ohem_)>fg_rois_per_image:
                    fg_inds = np.argsort(-loss_ohem_[:len(rcnn_smooth_l1_ohem_)])[:fg_rois_per_image]
                # pdb.set_trace()
                ohem_ind = (np.argsort(-loss_ohem_[len(rcnn_smooth_l1_ohem_):])+len(rcnn_smooth_l1_ohem_))[:min(rois_per_image-len(fg_inds),3*len(fg_inds))]
                ohem_ind = np.hstack([fg_inds, ohem_ind])
                batch_top_rois=batch_top_rois[ohem_ind]
                batch_fuse_labels=batch_fuse_labels[ohem_ind]
                batch_fuse_targets=batch_fuse_targets[ohem_ind]
            else:
                batch_top_rois, batch_fuse_labels, batch_fuse_targets  = \
                 rcnn_target(  batch_proposals, batch_gt_labels, batch_gt_top_boxes, batch_gt_boxes3d )
            
            batch_rois3d	 = project_to_roi3d    (batch_top_rois)
            batch_front_rois = project_to_front_roi(batch_rois3d  )
            batch_rgb_rois   = project_to_rgb_roi  (batch_rois3d, rgb_shape[1], rgb_shape[0])



            ##debug gt generation
            if vis and iter%iter_debug==0:
                top_image = top_imgs[idx]
                rgb       = rgbs[idx]

                img_gt     = draw_rpn_gt(top_image, batch_gt_top_boxes, batch_gt_labels)
                img_label  = draw_rpn_labels (img_gt, anchors, batch_top_inds, batch_top_labels )
                img_target = draw_rpn_targets(top_image, anchors, batch_top_pos_inds, batch_top_targets)
                #imshow('img_rpn_gt',img_gt)
                imshow('img_anchor_label',img_label)
                #imshow('img_rpn_target',img_target)

                img_label  = draw_rcnn_labels (top_image, batch_top_rois, batch_fuse_labels )
                img_target = draw_rcnn_targets(top_image, batch_top_rois, batch_fuse_labels, batch_fuse_targets)
                #imshow('img_rcnn_label',img_label)
                if vis :
                    imshow('img_rcnn_target',img_target)


                img_rgb_rois = draw_boxes(rgb, batch_rgb_rois[:,1:5], color=(255,0,255), thickness=1)
                if vis :
                    imshow('img_rgb_rois',img_rgb_rois)
                    cv2.waitKey(1)

            ## run classification and regression loss -----------
            fd2={
				**fd1,

                top_images: batch_top_images,
                front_images: batch_front_images,
                rgb_images: batch_rgb_images,

				top_rois:   batch_top_rois,
                front_rois: batch_front_rois,
                rgb_rois:   batch_rgb_rois,

                top_inds:     batch_top_inds,
                top_pos_inds: batch_top_pos_inds,
                top_labels:   batch_top_labels,
                top_targets:  batch_top_targets,

                fuse_labels:  batch_fuse_labels,
                fuse_targets: batch_fuse_targets,
            }
            _, rcnn_probs, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss = \
               sess.run([solver_step, fuse_probs, top_cls_loss, top_reg_loss, fuse_cls_loss, fuse_reg_loss],fd2)

            speed=time.time()-start_time
            log.write('%5.1f   %5d    %0.4fs   %0.4f   |   %0.5f   %0.5f   |   %0.5f   %0.5f  \n' %\
				(epoch, iter, speed, rate, batch_top_cls_loss, batch_top_reg_loss, batch_fuse_cls_loss, batch_fuse_reg_loss))

            # debug: ------------------------------------

            if vis and iter%iter_debug==0:
                top_image = top_imgs[idx]
                rgb       = rgbs[idx]

                batch_top_probs, batch_top_scores, batch_top_deltas  = \
                    sess.run([ top_probs, top_scores, top_deltas ],fd2)

                batch_fuse_probs, batch_fuse_deltas = \
                    sess.run([ fuse_probs, fuse_deltas ],fd2)
                # pdb.set_trace()
                #batch_fuse_deltas=0*batch_fuse_deltas #disable 3d box prediction
                probs, boxes3d = rcnn_nms(batch_fuse_probs, batch_fuse_deltas, batch_rois3d, threshold=0.05)

				## show rpn(top) nms
                img_rpn     = draw_rpn    (top_image, batch_top_probs, batch_top_deltas, anchors, inside_inds)
                img_rpn_nms = draw_rpn_nms(img_gt, batch_proposals, batch_proposal_scores)
                #imshow('img_rpn',img_rpn)
                if vis :
                    imshow('img_rpn_nms',img_rpn_nms)
                    cv2.waitKey(1)

                ## show rcnn(fuse) nms
                img_rcnn     = draw_rcnn (top_image, batch_fuse_probs, batch_fuse_deltas, batch_top_rois, batch_rois3d,darker=1)
                img_rcnn_nms = draw_rcnn_nms(rgb, boxes3d, probs)
                if vis :
                    imshow('img_rcnn',img_rcnn)
                    imshow('img_rcnn_nms',img_rcnn_nms)
                    cv2.waitKey(0)
            if (iter)%10==0:
                summary = sess.run(merged,fd2)
                train_writer.add_summary(summary, iter)
            # save: ------------------------------------
            if (iter)%5000==0 and (iter!=0):
                saver.save(sess, out_dir + '/check_points/snap_RVD_new_lidar_%06d.ckpt'%iter)  #iter

            idx=idx+1






## main function ##########################################################################

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_train()