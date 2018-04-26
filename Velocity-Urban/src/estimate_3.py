from intersect import get_intersect, read_line
import os
import cv2
import numpy as np
import time
import delta
import multiprocessing
import matplotlib.pyplot as plt

PATH_DIST = '../lines'
PATH_VID = '../video'
PATH_RES = '../video_estimate'
PATH_INFO = '../reidentify'
PATH_VEC = '../velocity'
ALPHA = 0.75
IS_VISUALIZE = False

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

def read_dist(video_name):
    video_name = video_name.split('.')[0]
    dist = {}
    file =  open('%s/%s/dis.txt' % (PATH_DIST, video_name), 'r')
    for line in file:
        [x, y, z] = line.split()
        dist['%s_%s' % (x, y)] = float(z)
        dist['%s_%s' % (y, x)] = float(z)
    return dist

def find_first_frame(frame_id, collision, lid):
    for i, fid in enumerate(frame_id):
        if (collision[i] == lid):
            return fid
    return -1

def find_last_frame(frame_id, collision, lid):
    N = frame_id.shape[0]
    for i in xrange(N):
        if (collision[N - i - 1] == lid):
            return frame_id[N - i - 1]
    return -1

def assign_velocity(frame, pairs, v, mean):
    for i, pair in enumerate(pairs):
        if pair[0] <= frame and frame <= pair[1]:
            return v[i]
    return mean

def get_velocity_one_object(obj, collision, lines, linenames, lineidx, dist, find):
    N = lines.shape[0]
    v = np.inf * np.ones((N - 1, ))
    velocity = np.inf * np.ones((obj.shape[0], ))
    mean = []
    pairs = []
    for i in xrange(1, N):
        first = find(obj[:, 0], collision, lineidx[i - 1])
        last  = find(obj[:, 0], collision, lineidx[i])
        if (first == -1 or last == -1):
            continue
        pairs.append((first, last))
        s = '%s_%s' % (linenames[i - 1], linenames[i])
        s = s.replace(" ", "")
        v[i - 1] = ALPHA * dist[s] * 30 * 0.681818 / (last - first)
        mean.append(v[i - 1])

    mean = np.inf if len(mean) == 0 else np.mean(mean)
    for i in xrange(N - 1):
        v[i] = mean if v[i] == np.inf else v[i]
    for i in xrange(obj.shape[0]):
        velocity[i] = assign_velocity(obj[i][0], pairs, v, mean)

    return velocity

def get_velocity_side(data, collision, side, lines, linenames, lineidx, dist, find):
    num_obj = int(np.max(data[:, 2]))
    velocity = np.inf * np.ones((data.shape[0], ))
    for obj_id in xrange(1, num_obj + 1):
        index = np.where(data[:, 2] == obj_id)[0]
        velocity[index] = get_velocity_one_object(data[index], collision[index], lines, linenames, lineidx, dist, find)
    return velocity

def smooth(info, velocity, side):
    hori = velocity[side == 2]
    vert = velocity[side < 2]
    hori = hori[hori != np.inf]
    vert = vert[vert != np.inf]
    hori = hori[hori != 0]
    vert = vert[vert != 0]

    meanh = np.mean(hori)
    meanv = np.mean(vert)
    mean  = np.mean(np.append(hori, vert))

    print 'Mean Hori', meanh
    print 'Mean Vert', meanv
    print 'Mean Total', mean


    for i, obj in enumerate(info):
        if velocity[i] == np.inf:
            if side[i] == 2:
                velocity[i] = meanh # velocity[i])
            if side[i] == 0 or side[i] == 1:
                velocity[i] = meanv # velocity[i])
            if side[i] == -1:
                velocity[i] = mean
        if velocity[i] > 65:
            velocity[i] = 65

    return velocity, mean

def get_velocity(video_name, data, lines, linenames, dist):
    left = []
    right = []
    hor = []
    for i in xrange(linenames.shape[0]):
        if linenames[i][0] == 'l':
            left.append(i)
        elif linenames[i][0] == 'r':
            right.append(i)
        else:
            hor.append(i)

    left = np.array(left)
    right = np.array(right)
    hor = np.array(hor)

    collision = np.zeros((data.shape[0], ))
    velocity  = np.inf * np.ones((data.shape[0], ))

    side, collision = get_intersect(video_name, lines, linenames=linenames)


    index = np.where(side == 0)[0]
    if len(index) > 0:
        velocity[index] = get_velocity_side(data[index], collision[index], 0, lines[right], linenames[right], right, dist, find_last_frame)
    index = np.where(side == 1)[0]
    if len(index) > 0:
        velocity[index] = get_velocity_side(data[index], collision[index], -1, lines[left], linenames[left], left, dist, find_last_frame)

    index = np.where(side == 2)[0]
    if len(index) > 0:
        velocity[index] = get_velocity_side(data[index], collision[index], -1, lines[hor], linenames[hor], hor, dist, find_last_frame)


    num_obj = int(np.max(data[:, 2]))
    now = np.inf * np.ones((num_obj + 1, ))
    for idx, obj in enumerate(data):
        obj_id = int(obj[2])
        if (velocity[idx] == np.inf):
            if now[obj_id] == np.inf:
                j = idx + 1
                while (j < data.shape[0]):
                    if (int(data[j, 2]) == obj_id) and (velocity[j] != np.inf):
                        now[obj_id] = velocity[j]
                        break
                    j = j + 1
            velocity[idx] = now[obj_id]
        else:
            now[obj_id] = velocity[idx]

    velocity = np.abs(velocity)

    return side, velocity, collision

def process(video_name):
    duration = time.time()
    video_name = video_name.split('.')[0]

    print 'Process video %s.mp4' % video_name

    lines, linecolors, linenames = read_line(video_name)
    data = np.load('%s/info_%s.npy' % (PATH_INFO, video_name))
    dist = read_dist(video_name)
    side, velocity, collision = get_velocity(video_name, data, lines, linenames, dist)
    status = delta.process(video_name, 10)
    assert status.shape[0] == velocity.shape[0]
    for i in xrange(status.shape[0]):
        if status[i] == 0:
            velocity[i] = 0

    velocity, mean = smooth(data, velocity, side)

    for i in xrange(status.shape[0]):
        if (status[i] == 1 and velocity[i] == 0):
            velocity[i] = mean

    np.save('%s/velocity_%s' % (PATH_VEC, video_name), velocity)
    np.save('%s/status_%s' % (PATH_VEC, video_name), status)

    if not IS_VISUALIZE:
        return 0

    # Visualiztion
    video = cv2.VideoCapture(PATH_VID + '/' + video_name + '.mp4')
    output = cv2.VideoWriter(PATH_RES + '/' + video_name + '.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (1920, 1080))
    idx = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        idx = idx + 1
        index = np.where(data[:,0] == idx)[0]
        for i in index:
            box = data[i, 3:7].astype(np.int)
            obj_id = int(data[i, 2])
            v = 'unknow mph' if velocity[i] == np.inf else '%0.2f mph' % velocity[i]
            text = str(obj_id).zfill(5) + ' ' + v
            color = (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1)
            cv2.putText(frame, v, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        output.write(frame)
    output.release()
    video.release()
    duration = time.time() - duration

    print 'Process video %s.mp4 takes %f second' % (video_name, duration)

if __name__ == '__main__':
    video_names = os.listdir(PATH_VID)
    video_names = [video_name for video_name in video_names if video_name[:4] == 'Loc3']
    video_names.sort()
    for video_name in video_names:
        process(video_name)


