from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *
import pdb


# gt_boxes    : (x1,y1,  x2,y2  label)  #projected 2d
# gt_boxes_3d : (x1,y1,z1,  x2,y2,z2,  ....    x8,y8,z8,  label)


def rcnn_target(rois, gt_labels, gt_boxes, gt_boxes3d):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = int(min(rois_per_image - fg_rois_per_this_image, 3*fg_rois_per_this_image))
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)

    return rois, labels, targets

# def project_to_rgb_roi(rois3d, width, height):
#     num  = len(rois3d)
#     rois = np.zeros((num,5),dtype=np.int32)
#     projections = box3d_to_rgb_projections(rois3d)
#     for n in range(num):
#         qs = projections[n]
#         minx = np.min(qs[:,0])
#         maxx = np.max(qs[:,0])
#         miny = np.min(qs[:,1])
#         maxy = np.max(qs[:,1])
#         minx = np.maximum(np.minimum(minx, width - 1), 0)
#         maxx = np.maximum(np.minimum(maxx, width - 1), 0)
#         miny = np.maximum(np.minimum(miny, height - 1), 0)
#         maxy = np.maximum(np.minimum(maxy, height - 1), 0)
#         rois[n,1:5] = minx,miny,maxx,maxy
#     return rois

def rcnn_target_2d(rois, gt_labels, gt_boxes, gt_boxes3d, gt_boxes2d, width, height):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = int(min(rois_per_image - fg_rois_per_this_image, 3*fg_rois_per_this_image))
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    gt_boxes2d = gt_boxes2d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        et_boxes2d = project_to_rgb_roi(et_boxes3d, width, height)
        targets_2d = box_transform_2d(et_boxes2d, gt_boxes2d)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)

    return rois, labels, targets, targets_2d

def rcnn_target_2d_z(rois, gt_labels, gt_boxes, gt_boxes3d, gt_boxes2d, width, height, proposals_z, gen_top_rois, gen_rois3D):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois,gen_top_rois, np.hstack((zeros, gt_boxes))))
    rois3D = top_z_to_box3d(rois[:,1:5],proposals_z)
    extended_rois3D = np.vstack((rois3D, gen_rois3D, gt_boxes3d))

    # extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    # rois3D = top_z_to_box3d(rois[:,1:5],proposals_z)
    # extended_rois3D = np.vstack((rois3D, gt_boxes3d))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    gt_nums = len(gt_boxes3d)
    # gt_max_overlaps = overlaps[:-gt_nums].argmax(axis=0)
    # gt_argmax_overlaps = max_overlaps[gt_max_overlaps]
    # fg_max_inds = gt_max_overlaps[np.where(gt_argmax_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]]
    # pdb.set_trace()
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    # fg_inds = np.setdiff1d(fg_inds, fg_max_inds)
    # pdb.set_trace()
    fg_inds_assigments = gt_assignment[fg_inds]

    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))

    nums_every_targets = 0.8*fg_rois_per_this_image//gt_nums
    inds_every_target=[]

    for i in range(gt_nums):
        target_inds = fg_inds[np.where(fg_inds_assigments==i)[0]]
        # pdb.set_trace()
        if target_inds.size > 0:
            target_inds = np.random.choice(target_inds, size=int(min(nums_every_targets, target_inds.size)), replace=False)
            inds_every_target.append(target_inds.reshape(-1,1))
    inds_every_target=np.vstack(inds_every_target)
    fg_inds = np.setdiff1d(fg_inds, inds_every_target)
    # pdb.set_trace()
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image-inds_every_target.size, replace=False)
    fg_inds = np.union1d(fg_inds.reshape(-1,1),inds_every_target)
    # fg_inds = np.union1d(fg_inds,fg_max_inds)

    print('nums of pos :%d'%len(fg_inds))
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    bg_rois_per_this_image = int(min(rois_per_image - fg_rois_per_this_image, 3*fg_rois_per_this_image))
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds.reshape(-1), bg_inds)
    rois   = extended_rois[keep]
    rois3D = extended_rois3D[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[len(fg_inds):] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    gt_boxes2d = gt_boxes2d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        et_boxes2d = project_to_rgb_roi(et_boxes3d, width, height)
        targets_2d = box_transform_2d(et_boxes2d, gt_boxes2d)
        # targets = box3d_transform(et_boxes3d, gt_boxes3d)
        # targets = box3d_transform(et_boxes3d, gt_boxes3d)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)

    return rois, labels, targets, targets_2d, rois3D,np.arange(len(fg_inds))


def rcnn_target_2d_z_ohem(rois, gt_labels, gt_boxes, gt_boxes3d, gt_boxes2d, width, height, proposals_z,batch_gt_boxesZ):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    rois3D = top_z_to_box3d(rois[:,1:5],proposals_z)

    # extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    # extended_rois3D = np.vstack((rois3D, gt_boxes3d))
    # extended_proposal_z=np.vstack((proposals_z, batch_gt_boxesZ))

    extended_rois = rois
    extended_rois3D = rois3D
    extended_proposal_z=proposals_z

    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    gt_nums = 0#len(gt_boxes3d)
    # gt_max_overlaps = overlaps[:-gt_nums].argmax(axis=0)
    # gt_argmax_overlaps = max_overlaps[gt_max_overlaps]
    # fg_max_inds = gt_max_overlaps[np.where(gt_argmax_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]]

    # pdb.set_trace()
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    # fg_inds = np.setdiff1d(fg_inds, fg_max_inds)
    fg_rois_per_this_image = int(min(fg_rois_per_image-gt_nums, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
    # fg_inds = np.union1d(fg_inds,fg_max_inds)
    print('nums of pos :%d'%len(fg_inds))
    fg_rois_per_this_image = int(fg_inds.size)
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    # bg_rois_per_this_image = int(min(rois_per_image - fg_rois_per_this_image, 3*fg_rois_per_this_image))
    # bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    # if bg_inds.size > 0:
    #     bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    rois3D = extended_rois3D[keep]
    rois_proposal_z=extended_proposal_z[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    gt_boxes2d = gt_boxes2d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        et_boxes2d = project_to_rgb_roi(et_boxes3d, width, height)
        targets_2d = box_transform_2d(et_boxes2d, gt_boxes2d)
        # targets = box3d_transform(et_boxes3d, gt_boxes3d)
        # targets = box3d_transform(et_boxes3d, gt_boxes3d)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)
    return rois, labels, targets, targets_2d, rois3D, rois_proposal_z


def rcnn_target_ohem_2d(rois, gt_labels, gt_boxes, gt_boxes3d, gt_boxes2d, width, height):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    print('nums of pos :%d'%len(fg_inds))
    fg_rois_per_this_image = int(fg_inds.size)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    gt_boxes2d = gt_boxes2d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)

        et_boxes2d = project_to_rgb_roi(et_boxes3d, width, height)
        targets_2d = box_transform_2d(et_boxes2d, gt_boxes2d)


    return rois, labels, targets, targets_2d



def rcnn_target_ohem(rois, gt_labels, gt_boxes, gt_boxes3d):

    # Include "ground-truth" in the set of candidate rois
    rois = rois.reshape(-1,5)  # Proposal (i, x1, y1, x2, y2) coming from RPN
    num           = len(gt_boxes)
    zeros         = np.zeros((num, 1), dtype=np.float32)
    extended_rois = np.vstack((rois, np.hstack((zeros, gt_boxes))))
    assert np.all(extended_rois[:, 0] == 0), 'Only single image batches are supported'


    rois_per_image    = CFG.TRAIN.RCNN_BATCH_SIZE
    fg_rois_per_image = np.round(CFG.TRAIN.RCNN_FG_FRACTION * rois_per_image)

    # overlaps: (rois x gt_boxes)
    overlaps = box_overlaps(
        np.ascontiguousarray(extended_rois[:,1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float)
    )
    max_overlaps  = overlaps.max(axis=1)
    gt_assignment = overlaps.argmax(axis=1)
    labels        = gt_labels[gt_assignment]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= CFG.TRAIN.RCNN_FG_THRESH_LO)[0]
    fg_rois_per_this_image = int(fg_inds.size)
    # fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # if fg_inds.size > 0:
    #     fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < CFG.TRAIN.RCNN_BG_THRESH_HI) &
                       (max_overlaps >= CFG.TRAIN.RCNN_BG_THRESH_LO))[0]
    # bg_rois_per_this_image = int(min(rois_per_image - fg_rois_per_this_image, 3*fg_rois_per_this_image))
    # bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    # if bg_inds.size > 0:
    #     bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)


    # The indices that we're selecting (both fg and bg)
    keep   = np.append(fg_inds, bg_inds)
    rois   = extended_rois[keep]
    labels = labels[keep]                # Select sampled values from various arrays:
    labels[fg_rois_per_this_image:] = 0  # Clamp la bels for the background RoIs to 0


    gt_boxes3d = gt_boxes3d[gt_assignment[keep]]
    et_boxes=rois[:,1:5]
    if gt_boxes3d.shape[1:]==gt_boxes.shape[1:]:
        #normal image faster-rcnn .... for debug
        targets = box_transform(et_boxes, gt_boxes3d)
        #targets = targets / np.array(CFG.TRAIN.RCNN_box_NORMALIZE_STDS)  # this is for each box
    else:
        et_boxes3d = top_box_to_box3d(et_boxes)
        targets = box3d_transform(et_boxes3d, gt_boxes3d)
        #exit(0)

    return rois, labels, targets




def draw_rcnn_labels(image, rois,  labels, darker=0.7):
    is_print=0

    ## draw +ve/-ve labels ......
    boxes = rois[:,1:5]
    labels = labels.reshape(-1)

    fg_label_inds = np.where(labels != 0)[0]
    bg_label_inds = np.where(labels == 0)[0]
    num_pos_label = len(fg_label_inds)
    num_neg_label = len(bg_label_inds)
    if is_print: print ('rcnn label : num_pos=%d num_neg=%d,  all = %d'  %(num_pos_label, num_neg_label,num_pos_label+num_neg_label))

    img_label = image.copy()*darker
    if 1:
        for i in bg_label_inds:
            a = boxes[i]
            cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (32,32,32), 1)
            cv2.circle(img_label,(a[0], a[1]),2, (32,32,32), -1)

    for i in fg_label_inds:
        a = boxes[i]
        cv2.rectangle(img_label,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)
        cv2.circle(img_label,(a[0], a[1]),2, (255,0,255), -1)

    return img_label

def draw_rcnn_targets(image, rois, labels,  targets, darker=0.7):
    is_print=1

    #draw +ve targets ......
    boxes = rois[:,1:5]

    fg_target_inds = np.where(labels != 0)[0]
    num_pos_target = len(fg_target_inds)
    if is_print: print ('rcnn target : num_pos=%d'  %(num_pos_target))

    img_target = image.copy()*darker
    for n,i in enumerate(fg_target_inds):
        a = boxes[i]
        cv2.rectangle(img_target,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)

        if targets.shape[1:]==(4,):
            t = targets[n]
            b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
            b = b.reshape(4)
            cv2.rectangle(img_target,(b[0], b[1]), (b[2], b[3]), (255,255,255), 1)

        if targets.shape[1:]==(8,3):
            t = targets[n]
            a3d = top_box_to_box3d(a.reshape(1,4))
            b3d = box3d_transform_inv(a3d, t.reshape(1,8,3))
            #b3d = b3d.reshape(1,8,3)
            img_target = draw_box3d_on_top(img_target, b3d, darken=1)

    return img_target





