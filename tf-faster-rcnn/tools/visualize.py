import cv2
import os
import numpy as np
import time

PATH_VIDEO = '../video'
PATH_INFO = '../result'
PATH_SMALL = '../small_info'
PATH_ORI = '../ori_info'
PATH_VIS = 'video_visualize'
THRESH = 0.6

def overlap(anchor, other):
    dx = min(anchor[2], other[2]) - max(anchor[0], other[0])
    dy = min(anchor[3], other[3]) - max(anchor[1], other[1])
    area = 0.0
    if (dx > 0) and (dy > 0):
        area = dx * dy
    anchor_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    overlap_rate = area / max(anchor_area, other_area)
    return overlap_rate

def visualize(video, info, writer):
    for i in xrange(1, 1801):
        _, frame = video.read()
        boxes = info[info[:, 0] == i]
        for box in boxes:
            box = box[2:].astype(np.int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        writer.write(frame)

def prune(info, thresh=0.6):
    ans = []
    for idx in xrange(1, 1801):
        box = info[info[:,0] == idx]
        N = box.shape[0]
        mark = -1.0 * np.ones((N, ))
        count = 0
        for i in xrange(N):
            if (mark[i] != -1):
                continue
            for j in xrange(i, N):
                rate = overlap(box[i,:4], box[j, :4])
                if rate >= thresh:
                    mark[j] = count
            count = count + 1

        for i in xrange(count):
            newbox = []
            b = box[mark == i]
            newbox.append(np.mean(b[:,0]))
            newbox.append(np.mean(b[:,1]))
            newbox.append(np.mean(b[:,2]))
            newbox.append(np.mean(b[:,3]))
            newbox.append(np.mean(b[:,4]))
            newbox.append(idx)
            ans.append(newbox)

    return np.array(ans)



def process(video_name):
    path_info = '%s/info_%s.npy' % (PATH_INFO, video_name)
    path_small_info = '%s/info_%s.npy' % (PATH_SMALL, video_name)
    path_ori_info = '%s/info_%s.npy' % (PATH_ORI, video_name)
    path_video = '%s/%s.mp4' % (PATH_VIDEO, video_name)
    path_vis = '%s/%s.mp4' % (PATH_VIS, video_name)
    info = np.load(path_info)
    small_info = np.load(path_small_info)
    ori_info = np.load(path_ori_info)
    #video = cv2.VideoCapture(path_video)
    #writer = cv2.VideoWriter(path_vis, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))

    res = []
    for idx in xrange(1, 1801):
        bbox1 = info[info[:, 0] == idx]
        bbox2 = ori_info[ori_info[:, -1] == idx]
        bbox3 = small_info[small_info[:, -1] == idx]
        for box in bbox1:
            coor = box[-4:].astype(np.int)
            res.append(box[:2].tolist() + box[-4:].tolist())
            assert len(res[-1]) == 6
        for box in bbox2:
            coor = box[:4].astype(np.int)
            res.append([box[-1], box[-2]] + box[:4].tolist())
            assert len(res[-1]) == 6

        for box in bbox3:
            box[:4] = box[:4] / 3
            box[0] = box[0] + 480
            box[2] = box[2] + 480
            coor = box[:4].astype(np.int)
            res.append([box[-1], box[-2]] + box[:4].tolist())
            assert len(res[-1]) == 6

    res = prune(np.array(res))
    #visualize(video, res, writer)
    np.save('info/info_%s' % video_name, res)
    #video.release()
    #writer.release()

    return np.array(res)

if __name__ == '__main__':
    list_file = os.listdir(PATH_VIDEO)
    video_names = [video_name.split('.')[0] for video_name in list_file if video_name[:4] == 'Loc3']
    video_names.sort()
    for video_name in video_names:
        print 'Process video %s.mp4' % video_name
        duration = time.time()
        process(video_name)
        duration = time.time() - duration
        print 'Process video %s.mp4 take %f second' % (video_name, duration)
