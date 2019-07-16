import numpy as np
import cv2
import os
from PIL import Image

#video_path = 'E:/golf-analyze/스윙분석/Golf_Swing_JinseonHan.mp4'
video_path = 'E:/Golf/data/스윙분석/Golf_Swing_HyeseonKim.mp4'
#video_path = 'E:/golf-analyze/스윙분석/Golf_Swing_Slow_2person.mp4'
vid = cv2.VideoCapture(video_path)
'''
out_vid = cv2.VideoWriter('E:/golf-analyze/ppt/vid9.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          vid.get(cv2.CAP_PROP_FPS),
                          (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
'''

start = 0
last = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
sampling = 10

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
    if frame_cnt ==start:
        ret, frame = vid.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        motion = cv2.cvtColor(np.zeros_like(frame), cv2.COLOR_BGR2BGRA)
        motion[:, :, :] = 0
        frame_cnt += 1
        continue
    elif frame_cnt == last:
        break

    if frame_cnt % sampling != 0:
        frame_cnt += 1
        continue

    visual = frame.copy()
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.GaussianBlur(next_gray, (5, 5), 0)
    dif = cv2.absdiff(gray, next_gray)
    _, thr = cv2.threshold(dif, 10, 255, cv2.THRESH_BINARY)

    blur = cv2.blur(frame, (30, 30))
    blur[thr != 0] = frame[thr != 0]
    # edge = cv2.Canny(blur, 50, 150)

    # contours, hierachy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_mask = np.zeros_like(gray)
    for cnt in contours:
        cv2.drawContours(image=visual, contours=[cnt], contourIdx=-1, color=255, thickness=-1)
        cv2.drawContours(image=action_mask, contours=[cnt], contourIdx=-1, color=255, thickness=-1)

    # fill only new area
    motion[:, :, 3] = cv2.bitwise_or(motion[:, :, 3], action_mask)
    motion[:, :, :3][action_mask != 0] = frame[action_mask != 0]
    act=cv2.bitwise_and(frame,frame,mask=action_mask)
    cv2.imshow('act',act)
    cv2.imshow('motion',motion)
    #out_vid.write(cv2.cvtColor(motion, cv2.COLOR_BGRA2BGR))
    frame=next_frame
    gray=next_gray
    frame_cnt += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

motion[:,:,:3][motion[:,:,3]==0]=last_frame[motion[:,:,3]==0]
out = cv2.cvtColor(motion, cv2.COLOR_BGRA2BGR)
#out_vid.write(out)
cv2.imshow('out', out)
cv2.waitKey(0)
out = cv2.addWeighted(start_frame, 0.5, out, 0.5, 0)
#out_vid.write(out)
#out = cv2.addWeighted(last_frame, 0.5, motion, 0.8, 0)

cv2.imshow('out', out)
cv2.waitKey(0)

cv2.destroyAllWindows()
