import numpy as np
import cv2
import os
from PIL import Image

video_path = 'E:/Golf/data/스윙분석/Golf_Swing_JinseonHan.mp4'
video_path = 'E:/Golf/data/스윙분석/Golf_Swing_HyeseonKim.mp4'
video_path = 'E:/Golf/data/스윙분석/Golf_Swing_Slow_2person.mp4'
name = os.path.splitext(os.path.basename(video_path))[0]
seg_path = 'E:/golf-analyze/segmentation/seg-{}'.format(name)

vid = cv2.VideoCapture(video_path)

'''
out_vid = cv2.VideoWriter('E:/Golf/data/nohuman2.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          vid.get(cv2.CAP_PROP_FPS),
                          (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
'''

start = 0
last = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
sampling = 2
last=last-sampling

vid.set(cv2.CAP_PROP_POS_FRAMES, start)
_, start_frame = vid.read()

vid.set(cv2.CAP_PROP_POS_FRAMES, last)
_, last_frame = vid.read()

print('interval {} - {} frames'.format(start, last))
vid.set(cv2.CAP_PROP_POS_FRAMES, start)

frame_cnt = start
while vid.isOpened():
    ret, next_frame = vid.read()
    if not ret: break
    if frame_cnt == start:
        ret, frame = vid.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        motion = cv2.cvtColor(np.zeros_like(frame), cv2.COLOR_BGR2BGRA)
        motion[:, :, :] = 0
        human = cv2.cvtColor(np.zeros_like(frame), cv2.COLOR_BGR2BGRA)
        human[:, :, :] = 0
        frame_cnt += 1
        continue
    elif frame_cnt == last:
        break

    visual = frame.copy()
    if frame_cnt % sampling == 0:
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.GaussianBlur(next_gray, (5, 5), 0)
        dif = cv2.absdiff(gray, next_gray)
        _, thr = cv2.threshold(dif, 10, 255, cv2.THRESH_BINARY)

        #blur = cv2.blur(frame, (30, 30))
        #blur[thr != 0] = frame[thr != 0]
        #edge = cv2.Canny(blur, 50, 150)

        #contours, hierachy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierachy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        action_mask = np.zeros_like(gray)
        for cnt in contours:
            # cv2.drawContours(image=visual, contours=[cnt], contourIdx=-1, color=255, thickness=-1)
            cv2.drawContours(image=action_mask, contours=[cnt], contourIdx=-1, color=255, thickness=-1)

        # motion accumulate - except human
        seg_file = os.path.join(seg_path, 'result_{}.png'.format(frame_cnt-sampling))
        seg_file2 = os.path.join(seg_path, 'result_{}.png'.format(frame_cnt))
        seg_file3 = os.path.join(seg_path, 'result_{}.png'.format(frame_cnt+sampling))
        if os.path.exists(seg_file) :
            seg = cv2.imread(seg_file, cv2.IMREAD_UNCHANGED)
            seg2 = cv2.imread(seg_file2, cv2.IMREAD_UNCHANGED)
            seg3 = cv2.imread(seg_file3, cv2.IMREAD_UNCHANGED)

            seg_m=cv2.bitwise_or(seg2[:,:,3],seg[:,:,3])
            seg_m = cv2.bitwise_or(seg_m, seg3[:, :, 3])
            cv2.imshow('seg_m', seg_m)
            human[:,:,3] = cv2.bitwise_or(human[:, :, 3], seg_m)
            stick = cv2.bitwise_and(cv2.bitwise_not(seg_m), action_mask)
            human[:,:,3][stick!=0]=0
            cv2.imshow('stick', stick)
            cv2.imshow('human', human[:,:,3])
            motion[:, :, 3] = cv2.bitwise_or(motion[:, :, 3], stick)
            motion[:, :, 3][human[:,:,3]!=0] = 0
            cv2.imshow('motion3', motion[:, :, 3])
            motion[:, :, :3][stick != 0] = frame[stick != 0]
        else:
            motion[:, :, 3] = cv2.bitwise_or(motion[:, :, 3], action_mask)
            motion[:, :, :3][action_mask != 0] = frame[action_mask != 0]

        act = cv2.bitwise_and(frame, frame, mask=action_mask)
        cv2.imshow('act', act)
        frame = next_frame
        gray = next_gray

    cv2.imshow('motion', motion)

    visual[:, :, :3][motion[:, :, 3] != 0] = motion[:, :, :3][motion[:, :, 3] != 0]
    #visual=cv2.addWeighted(visual,.3,frame,.7,-30)
    #visual=cv2.blur(visual,(3,3))qq
    cv2.imshow('visual', visual)
    

    #visual=cv2.addWeighted(visual,.5,motion[:,:,:3],.6,0)
    #visual = cv2.add(visual, motion_bg[:, :, :3],mask=motion_bg[:,:,3])

    # out_vid.write(cv2.cvtColor(motion, cv2.COLOR_BGRA2BGR))

    frame_cnt += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


while vid.isOpened():
    ret, frame = vid.read()
    if not ret: break
    visual=frame.copy()

    visual[:, :, :3][motion[:, :, 3] == 0] = frame[:, :, :3][motion[:, :, 3] == 0]
    visual[:, :, :3][motion[:, :, 3] != 0] = motion[:, :, :3][motion[:, :, 3] != 0]
    cv2.imshow('visual', visual)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

last_frame = frame
last = frame_cnt
'''
seg_file_start = os.path.join(seg_path, 'result_{}.png'.format(start)) + 'nonono'
seg_file_last = os.path.join(seg_path, 'result_{}.png'.format(last))
if os.path.exists(seg_file_last):
    seg = cv2.imread(seg_file_last, cv2.IMREAD_UNCHANGED)
    seg_m = seg[:, :, 3]
    motion[:, :, :3][seg_m != 0] = seg[:, :, :3][seg_m != 0]
    motion[:, :, 3] = cv2.bitwise_or(motion[:, :, 3], seg[:, :, 3])
    visual[:, :, :3][seg_m != 0] = seg[:, :, :3][seg_m != 0]

#motion[:, :, :3][motion[:, :, 3] == 0] = last_frame[motion[:, :, 3] == 0]
# motion[:, :, :3][motion[:, :, 3] == 0] = start_frame[motion[:, :, 3] == 0]
#motion = cv2.cvtColor(motion, cv2.COLOR_BGRA2BGR)
'''
cv2.imshow('motion', motion)
cv2.imshow('vis', visual)
# out_vid.write(out)
cv2.waitKey()
# out = cv2.addWeighted(start_frame, 0.3, motion, 0.7, 0)
# out_vid.write(out)


cv2.destroyAllWindows()
