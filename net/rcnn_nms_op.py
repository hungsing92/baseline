from net.common import *
from net.configuration import *
from net.processing.boxes import *
from net.processing.boxes3d import *
from net.utility.draw import *

from net.processing.cpu_nms import cpu_nms as nms


def draw_rcnn_berfore_nms(image, probs,  deltas, rois, rois3d, threshold=0.8):

    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>threshold)[0]

    #post processing
    rois   = rois[idx]
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]

    num = len(rois)
    for n in range(num):
        a   = rois[n,1:5]
        cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)


    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(rois[:,1:5],deltas)
        ## <todo>

    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        boxes3d  = regularise_box3d(boxes3d)
        draw_box3d_on_top(image,boxes3d)


#after nms : camera image
def draw_rcnn_nms_rgb(rgb, boxes3d, probs, darker=0.7):

    img_rcnn_nms = rgb.copy()*darker
    projections = box3d_to_rgb_projections(boxes3d)
    img_rcnn_nms = draw_rgb_projections(img_rcnn_nms,  projections, color=(255,255,255), thickness=1)

    return img_rcnn_nms

#after nms : lidar top image
def draw_rcnn_after_nms_top(image, boxes3d, probs):
    draw_box3d_on_top(image,boxes3d)



def draw_rcnn(image, probs,  deltas, rois, rois3d, threshold=0.1, darker=0.7):

    img_rcnn = image.copy()*darker
    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>threshold)[0]

    #post processing
    rois   = rois[idx]
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]

    num = len(rois)
    for n in range(num):
        a   = rois[n,1:5]
        cv2.rectangle(img_rcnn,(a[0], a[1]), (a[2], a[3]), (255,0,255), 1)


    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(rois[:,1:5],deltas)
        ## <todo>

    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        boxes3d  = regularise_box3d(boxes3d)
        img_rcnn = draw_box3d_on_top(img_rcnn,boxes3d)

    return img_rcnn



def draw_rcnn_nms(rgb, boxes3d, probs, darker=0.7):

    img_rcnn_nms = rgb.copy()*darker
    projections = box3d_to_rgb_projections(boxes3d)
    img_rcnn_nms = draw_rgb_projections(img_rcnn_nms,  projections, color=(255,255,255), thickness=1)

    return img_rcnn_nms




## temporay post-processing ....
## <todo> to be updated


def rcnn_nms( probs,  deltas,  rois3d,  threshold = 0.05):


    cls=1  # do for class-one only
    probs = probs[:,cls] #see only class-1
    idx = np.where(probs>0.96)[0]

    #post processing
    rois3d = rois3d[idx]
    deltas = deltas[idx,cls]
    probs  = probs [idx]

    if deltas.shape[1:]==(4,):
        boxes = box_transform_inv(priors,deltas)
        return probs,boxes


    if deltas.shape[1:]==(8,3):
        boxes3d  = box3d_transform_inv(rois3d, deltas)
        top_boxes=box3d_to_top_box(boxes3d)
        keep = nms(np.hstack((top_boxes, probs.reshape(-1,1))), threshold)
        boxes3d=boxes3d[keep]
        boxes3d  = regularise_box3d(boxes3d)

        return probs, boxes3d