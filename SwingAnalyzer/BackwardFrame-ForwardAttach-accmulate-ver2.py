import numpy as np
import cv2
import os
from PIL import Image

# video_path = 'E:/golf-analyze/스윙분석/Golf_Swing_JinseonHan.mp4'
#video_path = 'E:/Golf/data/스윙분석/Golf_Swing_HyeseonKim.mp4'
#video_path = 'E:/Golf/data/스윙분석/Golf_Swing_Slow_2person.mp4'
video_path = 'E:/Golf/data/프로스윙2/박결_드라이버_측면.mp4' #120-1200
vid = cv2.VideoCapture(video_path)
'''
out_vid = cv2.VideoWriter('E:/golf-analyze/ppt/vid9.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          vid.get(cv2.CAP_PROP_FPS),
                          (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
'''

start = 120
last = 1200#int(vid.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
sampling = 5

vid.set(cv2.CAP_PROP_POS_FRAMES, start)
_, start_frame = vid.read()

vid.set(cv2.CAP_PROP_POS_FRAMES, last)
_, last_frame = vid.read()

print('interval {} - {} frames'.format(start, last))
vid.set(cv2.CAP_PROP_POS_FRAMES, start)

frame_cnt = start
while vid.isOpened():
    ret, frame = vid.read()
    if not ret: break
    if frame_cnt == start:
        motion = cv2.cvtColor(np.zeros_like(frame), cv2.COLOR_BGR2BGRA)
        motion[:, :, :] = 0
    elif frame_cnt == last:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    visual = frame.copy()

    ret,next_frame=vid.read()
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.GaussianBlur(next_gray, (5, 5), 0)
    dif = cv2.absdiff(gray,next_gray)
    _, thr = cv2.threshold(dif, 5, 255, cv2.THRESH_BINARY)

    #blur = cv2.blur(frame, (30, 30))
    #blur[thr != 0] = frame[thr != 0]
    #edge = cv2.Canny(blur, 50, 100)
    #cv2.imshow('ed',edge)

    #contours, hierachy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_mask = np.zeros_like(gray)
    for cnt in contours:
        #cv2.drawContours(image=visual, contours=[cnt], contourIdx=-1, color=255, thickness=-1)
        cv2.drawContours(image=action_mask, contours=[cnt], contourIdx=-1, color=255, thickness=-1)
    # motion accumulate - simple
    motion[:, :, 3] = cv2.bitwise_or(motion[:, :, 3], action_mask)
    motion[:, :, :3][action_mask != 0] = frame[action_mask != 0]

    # motion accumulate - simple

    act = cv2.bitwise_and(frame, frame, mask=action_mask)
    #cv2.imshow('inter1',inter)
    cv2.imshow('act', act)
    cv2.imshow('motion', motion)


    #visual[:, :, :3][motion[:, :, 3] != 0] = motion[:, :, :3][motion[:, :, 3] != 0]
    #visual=cv2.add(motion[:,:,:3],visual)
    motion[:, :, :3][motion[:, :, 3] == 0] = frame[motion[:, :, 3] == 0]
    visual = cv2.addWeighted(motion[:, :, :3], .6, visual, .4, 0)
    #visual = cv2.addWeighted(motion[:, :, :3],.4, visual,.6,0)
    cv2.imshow('visual', visual)


    # out_vid.write(cv2.cvtColor(motion, cv2.COLOR_BGRA2BGR))
    vid.set(cv2.CAP_PROP_POS_FRAMES,frame_cnt+sampling)
    frame_cnt += sampling
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

motion[:, :, :3][motion[:, :, 3] == 0] = last_frame[motion[:, :, 3] == 0]
out = cv2.cvtColor(motion, cv2.COLOR_BGRA2BGR)
# out_vid.write(out)
cv2.imshow('out', out)
cv2.waitKey(0)
out = cv2.addWeighted(start_frame, 0.5, out, 0.5, 0)
# out_vid.write(out)
# out = cv2.addWeighted(last_frame, 0.5, motion, 0.8, 0)

cv2.imshow('out', out)
cv2.waitKey(0)

cv2.destroyAllWindows()
