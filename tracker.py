from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import math
prev_coords = {}
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, v, cls_id, pos_id) in bboxes:
        if cls_id in ['person']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, 'v:{}m/s ID-{} '.format(v , pos_id, cls_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image


def update_tracker(target_detector, image):
    global prev_coords
    new_faces = []
    _, bboxes = target_detector.detect(image)
    bbox_xywh = []
    confs = []
    clss = []
    if len(bboxes):
        for x1, y1, x2, y2, cls_id, conf in bboxes:
            obj = [
                int((x1+x2)/2), int((y1+y2)/2),
                x2-x1, y2-y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
            clss.append(cls_id)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, clss, image)

        bboxes2draw = []
        face_bboxes = []
        for value in list(outputs):
            x1, y1, x2, y2, cls_, track_id = value
            v = 0

            # 获取先前坐标信息，如果不存在，则将当前坐标作为初始值
            prev_x1, prev_y1, prev_x2, prev_y2, prev_v = prev_coords.get(track_id, (x1, y1, x2, y2, v))

            t = 4
            v = calculate_speed(x1, y1, x2, y2, prev_x1, prev_y1, prev_x2, prev_y2, t)
            if prev_v != 0:
                v = 0.3 * v + 0.7 * prev_v
            v = round(v, 1)
            # 更新先前坐标信息
            prev_coords[track_id] = (x1, y1, x2, y2, v)
            bboxes2draw.append(
                (x1, y1, x2, y2, v, cls_, track_id)
            )

    image = plot_bboxes(image, bboxes2draw)

    return image, new_faces, face_bboxes

def calculate_speed(x1, y1, x2, y2, prev_x1, prev_y1, prev_x2, prev_y2, t):
    # 计算两帧之间的位移
    delta_x = (x1 + x2 - prev_x1 - prev_x2) / 2
    delta_y = (y1 + y2 - prev_y1 - prev_y2) / 2

    # 计算欧几里得距离作为位移
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    speed = distance / t

    return speed

