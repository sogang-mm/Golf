import numpy as np
import cv2
import os
from PIL import Image

# video_path = 'E:/golf-analyze/스윙분석/Golf_Swing_JinseonHan.mp4'
video_path = 'E:/Golf/data/스윙분석/Golf_Swing_HyeseonKim.mp4'
# video_path = 'E:/golf-analyze/스윙분석/Golf_Swing_Slow_2person.mp4'
vid = cv2.VideoCapture(video_path)
'''
out_vid = cv2.VideoWriter('E:/golf-analyze/ppt/vid3.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          vid.get(cv2.CAP_PROP_FPS),
                          (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
'''
start = 0
last = vid.get(cv2.CAP_PROP_FRAME_COUNT)
sampling=1

vid.set(cv2.CAP_PROP_POS_FRAMES, start)
_, start_frame = vid.read()

vid.set(int(cv2.CAP_PROP_FRAME_COUNT) - 1, last)
_, last_frame = vid.read()

print('interval {} - {} frames'.format(start, last))
vid.set(cv2.CAP_PROP_POS_FRAMES, start)

frame_cnt = start
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    if frame_cnt==last:
        break
    if frame_cnt == start:
        prev_frame = frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        prev_motion = np.zeros_like(frame)
        frame_cnt += 1
        continue
    if frame_cnt%sampling!=0:
        frame_cnt += 1
        continue
    visual = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    dif = cv2.absdiff(gray, prev_gray)
    _, thr = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)

    blur = cv2.blur(frame, (30, 30))
    blur[thr != 0] = frame[thr != 0]
    # edge = cv2.Canny(blur, 50, 150)

    # contours, hierachy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_mask = np.zeros_like(gray)
    for cnt in contours:
        cv2.drawContours(image=visual, contours=[cnt], contourIdx=-1, color=255, thickness=-1)
        cv2.drawContours(image=action_mask, contours=[cnt], contourIdx=-1, color=255, thickness=-1)

    action = cv2.bitwise_and(frame, frame, mask=action_mask)
    # action=frame
    # motion=cv2.addWeighted(motion,.5,action,.9,0)
    motion = cv2.add(prev_motion, action, 0)

    cv2.imshow('motion', motion)
    #out_vid.write(motion)
    # cv2.imshow('edge', edge)

    prev_motion = motion
    prev_gray = gray
    frame_cnt += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imshow('out', motion)
cv2.waitKey(0)
mask=np.ones_like(gray)*255
mask=cv2.bitwise_and(motion[:,:,0],mask)
motion[mask==0]=last_frame[mask==0]
#out_vid.write(motion)
cv2.imshow('motion', motion)
cv2.waitKey(0)

out = cv2.addWeighted(last_frame, .5, start_frame,.5, 0)
out = cv2.addWeighted(out, 0.5, motion, 0.5, 0)
#out_vid.write(out)
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
