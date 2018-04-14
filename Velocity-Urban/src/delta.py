import os
import cv2
import numpy as np
import skvideo.io
import time

PATH_VID = '../video'
PATH_INFO = '../reidentify'
PATH_VIS = '../video_delta'
THRESH = 0.7
THRESH_COS = 0.9
WINDOW = 20

#0 stop
#1 run
#2 slow

def smooth(array, window=WINDOW, stride=WINDOW):
    length = window / 2
    N = array.shape[0]
    res = np.zeros((N, ))
    for i in xrange(1, N):
        v = [0, 0]
        for j in xrange(i - length, i + length):
            if j >= N:
                break
            if j < 0:
                break
            v[int(array[j])] = v[int(array[j])] + 1
        if (v[0] > v[1]):
            res[i] = 0
        else:
            res[i] = 1
    return res


def cosin_distance(pre,cur, box):
    a = pre[box[1]:box[3], box[0]:box[2]]
    b = cur[box[1]:box[3], box[0]:box[2]]
    sz = a.shape[0] * a.shape[1] * a.shape[2]
    a = np.reshape(a, sz) / 255.0
    b = np.reshape(b, sz) / 255.0
    #print a.shape, b.shape
    return np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

def overlap(anchor, other):
    dx = min(anchor[2], other[2]) - max(anchor[0], other[0])
    dy = min(anchor[3], other[3]) - max(anchor[1], other[1])
    area = 0.0
    if (dx > 0) and (dy > 0):
        area = dx * dy
    anchor_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    other_area = (other[2] - other[0]) * (other[3] - other[1])
    overlap_rate = area / max(anchor_area, other_area)
    #print overlap_rate
    return overlap_rate

def process(video_name, delta):
    path_info = '%s/info_%s.npy' % (PATH_INFO, video_name)
    path_video = '%s/%s.mp4' % (PATH_VID, video_name)
    path_vis = '/%s/%s.mp4' % (PATH_VIS, video_name)

    #[frame_id, score, x, y, z, y]
    info = np.load(path_info)
    status = -1 * np.ones((info.shape[0], ))
    num_obj = int(np.max(info[:, 2]))

    d = time.time()
    video = skvideo.io.vread(path_video)
    d = time.time() - d
    #print 'Load video %s takes %f second' % (video_name, d)

    for obj_id in xrange(1, num_obj + 1):
        index = np.where(info[:, 2] == obj_id)[0]
        N = index.shape[0]
        stat = -1 * np.ones((N, ))
        if N < delta:
            status[index] = stat
            continue
        rem = -1
        for i in xrange(delta - 1, N):
            pre = info[index[i - delta + 1]]
            cur = info[index[i]]
            rate = overlap(pre[3:], cur[3:])
            if (rate >= THRESH):
                stat[i-delta+1] = 0
            else:
                stat[i-delta+1] = 1
            rem = stat[i-delta+1]
        for i in xrange(N - delta + 1, N):
            stat[i] = rem

        stat = smooth(stat)

        for i in xrange(1, N):
            if stat[i] == 0:
                cur = video[int(info[index[i], 0]) - 1]
                pre = video[int(info[index[i -1], 0]) - 1]
                box = info[index[i], 3:].astype(np.int)
                dist = cosin_distance(cur, pre, box)
                if (dist >= THRESH_COS):
                    stat[i] = 0
                else:
                    stat[i] = 1
        status[index] = smooth(stat)
    return status


if __name__ == '__main__':
    video_name = 'Loc1_1.mp4'
    process(video_name, 10)






