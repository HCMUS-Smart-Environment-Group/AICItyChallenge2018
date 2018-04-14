import numpy as np
import os

PATH_VID = '../video'
PATH_SUB = 'submission'
PATH_INFO = 'result'
PATH_VEC = 'velocity'

LINE_SUB = '%d %d %d %d %d %d %d %.3f %f\n'
DEFAULT = 65.0

def process(output, idx, video_name):
    video_name = video_name.split('.')[0]
    print 'Process video %s.mp4' % video_name
    data = np.load('%s/info_%s.npy' % (PATH_INFO, video_name))
    velocity = np.load('%s/velocity_%s.npy' % (PATH_VEC, video_name))

    N = data.shape[0]
    assert data.shape[0] == velocity.shape[0]
    for i in xrange(N):
        frame_id = int(data[i, 0])
        conf = float(data[i, 1])
        obj_id = int(data[i, 2])
        xmin = int(data[i, 3])
        ymin = int(data[i, 4])
        xmax = int(data[i, 5])
        ymax = int(data[i, 6])
        v = float(velocity[i])
        assert velocity[i] != np.inf
        #if velocity[i] == np.inf:
        #    continue
        #if velocity[i] > 80 or velocity[i] == 0:
        #    continue
        assert 0 <= v and v <= 65
        res = LINE_SUB % (idx, frame_id, obj_id, xmin, ymin, xmax, ymax, abs(v), conf)
        output.write(res)

if __name__ == '__main__':
    video_name = 'Loc1_1.mp4'
    process(video_name)




