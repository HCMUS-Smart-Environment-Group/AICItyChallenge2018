import numpy as np
import os
import cv2
import time

PATH_INFO = '../reidentify'
PATH_VID  = '../video'
PATH_RES  = '../video_intersect'
PATH_LINE = '../lines'

def function(point, line):
    return (line[0][0] - line[1][0]) * (line[0][1] - point[1]) - (line[0][1] - line[1][1]) * (line[0][0] - point[0])

def in_mid(points, line):
    y1 = function(points[0], line)
    y2 = function(points[1], line)
    if y1 * y2 <= 0:
        return True
    else:
        return False

def is_intersect(line1, line2):
    return in_mid(line1, line2) and in_mid(line2, line1)

def bbox_intersect(box, line):
    line_boxes = [[(box[0], box[1]), (box[0], box[3])], [(box[0], box[1]), (box[2], box[1])],
                  [(box[2], box[1]), (box[2], box[3])], [(box[0], box[3]), (box[2], box[3])]]
    for line_box in line_boxes:
        if is_intersect(line, line_box):
            return True
    return False

def object_intersect(obj, lines, linenames):
    boxes = obj[:, 3:7]
    N = obj.shape[0]
    side = -1 * np.ones((N, ))
    collision = -1 * np.ones((N, ))

    for i, box in enumerate(boxes):
        tmp = []
        for lid, line in enumerate(lines):
            if bbox_intersect(box, line):
                tmp.append(lid)

        if len(tmp) > 0:
            side[i] = 0 if linenames[tmp[0]][0] == 'r' else  1
            collision[i] = tmp[0]

    return side, collision

def get_intersect(video_name, lines, linecolors=None, linenames=None, is_vis=False):
    video_name = video_name.split('.')[0]
    video = cv2.VideoCapture(PATH_VID + '/' + video_name + '.mp4')
    data = np.load(PATH_INFO + '/' + 'info_' + video_name + '.npy')

    sides = np.zeros((data.shape[0], ))
    collision = -1.0 * np.ones((data.shape[0], ))

    num_obj = int(np.max(data[:, 2]))
    for obj_id in xrange(1, num_obj + 1):
        index = np.where(data[:, 2] == obj_id)[0]
        side, coll = object_intersect(data[index], lines, linenames)
        sides[index] = side
        collision[index] = coll

    if not is_vis:
        return sides, collision

    output = cv2.VideoWriter(PATH_RES + '/' + video_name + '.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0, (1920, 1080))
    idx = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        idx = idx + 1
        index = np.where(data[:, 0] == idx)[0]
        for i, line in enumerate(lines):
            line = line.astype(np.int)
            cv2.line(frame, (line[0][0], line[0][1]) , (line[1][0], line[1][1]), linecolors[i], 3)
            cv2.putText(frame, str(linenames[i]).zfill(3), (line[0][0], line[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, linecolors[i], 3)

        for i in index:
            box = data[i, 3:7].astype(int)
            col = int(collision[i])
            obj_id = int(data[i, 2])
            color = (0, 0, 255)
            if col != -1:
                color = linecolors[col]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3 )
            cv2.putText(frame, str(obj_id).zfill(5), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        output.write(frame)

    video.release()
    output.release()

    return collision, collision

def random_color():
    return np.random.randint(0,255,(3)).tolist()

def read_line(video_name):
    data = open('%s/%s/line.txt' % (PATH_LINE, video_name), 'r').read().split('\n')
    N = len(data) / 2
    lines = np.zeros((N, 2, 2))
    linenames = []
    names = []
    linecolors = np.zeros((N, 3))
    for i in xrange(N):
        linecolors[i] = np.random.randint(0, 255, 3)
        linenames.append(data[2 * i])
        coor = [int(x) for x in data[2 * i + 1].split()]
        lines[i] = np.array([[coor[0], coor[1]], [coor[2], coor[3]]])

    linename = np.array(linenames)
    index = np.argsort(linenames)
    lines = lines[index]
    linenames = linename[index]
    linecolor = linecolors[index]
    revlines = {}
    return lines, linecolors, linenames


def process(video_name):
    video_name = video_name.split('.')[0]
    print 'Process video %s.mp4' % video_name
    duration = time.time()
    lines, linecolors, linenames = read_line(video_name)
    get_intersect(video_name, lines, linenames=linenames, linecolors=linecolors, is_vis=True)
    duration = time.time() - duration
    print 'Process %s.mp4 takes %f' % (video_name, duration)

if __name__ == '__main__':
    video_name = 'Loc1_1.mp4'
    process(video_name)
