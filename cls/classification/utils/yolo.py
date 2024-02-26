import time
import numpy as np
import cv2
from typing import List


def preprocess_yolo(img: np.ndarray, imgsz=(640, 640), fp16=False) -> np.ndarray:
    """Prepares input image before inference.

    :param img: input image (h0, w0, 3)
    :param imgsz: tuple of height and width, defaults to (640, 640)
    :param fp16: whether model use float16 or not, defaults to False

    :return: output np.ndarray (3, h, w)
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, imgsz)
    img = img.transpose(2, 0, 1)

    img = img.astype('float16') if fp16 else img.astype('float32')
    img /= 255  # 0 - 255 to 0.0 - 1.0

    return img


def postprocess_yolo(preds: List[np.ndarray],
                names: dict,
                orig_img_hw: tuple,
                img_hw: tuple,
                conf: float = 0.25,
                iou: float = 0.45,
                retina_masks: bool = False,
                ) -> List[List[np.ndarray]]:
    """Process outputs after inference.

    :param preds: List of 2 arrays: (B, N, CLS + 4 + M) and (B, M, H, W)
                  B = 1 - batch size
                  N - number of found objects
                  CLS - number of classes
                  M = 32 - mask dim
                  H - inference height
                  W - inference height

    :param names: dict of matched ids and class names ({0: cls_name0, 1: cls_name1, ...})
    :param orig_img_hw: original image height and width
    :param img_hw: inference image heigth and width
    :param conf: confidence threshold, defaults to 0.25
    :param iou: iou threshold, defaults to 0.45
    :param retina_masks: use retina mask, defaults to False

    :return: List of B lists of 2 arrays: detections (N, 6) and masks (N, H0, W0)
             Each row of detection array is (x1, y1, x2, y2, confidence, class_id)
             Each mask is binary array with values 1 or 0
    """

    # preds[0] = torch.tensor(preds[0])
    # preds[1] = torch.tensor(preds[1])

    p = non_max_suppression(preds[0],
                            conf,
                            iou,
                            agnostic=False,
                            max_det=300,
                            nc=len(names),
                            classes=None)

    if p[0].shape[0] == 0:
        return []

    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported

    for i, pred in enumerate(p):

        masks = process_mask(proto[i], pred[:, 6:], pred[:, :4], img_hw, upsample=True)  # HWC
        masks = masks.transpose(1, 2, 0)
        orig_sized_masks = []
        for i in range(masks.shape[2]):
            orig_sized_masks.append(np.expand_dims(cv2.resize(masks[:, :, i], orig_img_hw[::-1]), axis=0))
        masks = np.concatenate(orig_sized_masks, axis=0)
        pred[:, :4] = scale_boxes(img_hw, pred[:, :4], orig_img_hw)

        # segments = masks2segments(masks)
        bboxes = pred[:, :6]
        results.append([bboxes, masks])
    return results


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels


def masks2segments(masks, strategy='largest', epsilon=5):
    """
    It takes a list of masks(n,h,w) and returns a list of segments(n,xy)

    Args:
      masks (torch.Tensor): the output of the model, which is a tensor of shape (batch_size, 160, 160)
      strategy (str): 'concat' or 'largest'. Defaults to largest
      epsilon (float): Parameter specifying the approximation accuracy for Douglas-Peaker algorithm.
      This is the maximum distance between the original curve and its approximation.

    Returns:
      segments (List): list of segment masks
    """
    segments = []
    for x in masks.astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found

        # if c.shape[0] != 0:
        #     c = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        segments.append(c.astype('float32'))
    return segments


def non_max_suppression(
        prediction: np.ndarray,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].max(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(1, 0)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = np.concatenate((x, v), axis=0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask, _ = np.split(x, (4, 4 + nc, 4 + nc + nm), axis=1)
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf = cls.max(1, keepdims=True)
            j = cls.argmax(1, keepdims=True)
            
            x = np.concatenate((box, conf, j.astype(np.float32), mask), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def nms(boxes, scores, iou_thres):

    rows = len(boxes)
    
    sort_index = np.flip(scores.argsort())
    
    boxes = boxes[sort_index]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        condition = (iou > iou_thres)
        keep = keep & ~condition

    return np.nonzero(keep[sort_index.argsort()])[0]


def box_iou_batch(
	boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        
    return area_inter / (area_a[:, None] + area_b - area_inter)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


# def box_iou(box1, box2, eps=1e-7):
#     """
#     Calculate intersection-over-union (IoU) of boxes.
#     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#     Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

#     Args:
#         box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
#         box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
#         eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

#     Returns:
#         (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
#     """

#     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#     (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
#     inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

#     # IoU = inter / (area1 + area2 - inter)
#     return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    # if isinstance(boxes, torch.Tensor):  # faster individually
    #     boxes[..., 0].clamp_(0, shape[1])  # x1
    #     boxes[..., 1].clamp_(0, shape[0])  # y1
    #     boxes[..., 2].clamp_(0, shape[1])  # x2
    #     boxes[..., 3].clamp_(0, shape[0])  # y2
    # else:  # np.array (faster grouped)
    
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = sigmoid(masks_in @ protos.astype(np.float32).reshape(c, -1)).reshape(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = masks.transpose(1, 2, 0)
        masks = cv2.resize(masks, (iw, ih))
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        masks = masks.transpose(2, 0, 1)
        # masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    # return masks.gt_(0.5)
    return (masks > 0.5).astype('uint8')


def sigmoid(inp: np.ndarray):
    return 1 / (1 + np.exp(-inp))


# def process_mask_native(protos, masks_in, bboxes, shape):
#     """
#     It takes the output of the mask head, and crops it after upsampling to the bounding boxes.

#     Args:
#       protos (torch.Tensor): [mask_dim, mask_h, mask_w]
#       masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
#       bboxes (torch.Tensor): [n, 4], n is number of masks after nms
#       shape (tuple): the size of the input image (h,w)

#     Returns:
#       masks (torch.Tensor): The returned masks with dimensions [h, w, n]
#     """
#     c, mh, mw = protos.shape  # CHW
#     masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)
#     gain = min(mh / shape[0], mw / shape[1])  # gain  = old / new
#     pad = (mw - shape[1] * gain) / 2, (mh - shape[0] * gain) / 2  # wh padding
#     top, left = int(pad[1]), int(pad[0])  # y, x
#     bottom, right = int(mh - pad[1]), int(mw - pad[0])
#     masks = masks[:, top:bottom, left:right]

#     masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
#     masks = crop_mask(masks, bboxes)  # CHW
#     return masks.gt_(0.5)


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box

    Args:
      masks (torch.Tensor): [h, w, n] tensor of masks
      boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
      (torch.Tensor): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], [1, 2, 3], 1)  # x1 shape(n,1,1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
