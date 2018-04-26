import estimate_3 as loc3
import estimate_4 as loc4
import os

PATH_VID = '../video'
if __name__ == '__main__':
    video_names = os.listdir(PATH_VID)
    for video_name in video_names:
        if video_name[:4] == 'Loc3':
            loc3.process(video_name)
        elif video_name[:4] == 'Loc4':
            loc4.process(video_name)
