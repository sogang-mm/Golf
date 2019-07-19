import numpy as np
import cv2
import os
from PIL import Image

# video_path = 'E:/golf-analyze/스윙분석/Golf_Swing_JinseonHan.mp4'
# video_path = 'E:/Golf/data/스윙분석/Golf_Swing_HyeseonKim.mp4'
# video_path = 'E:/Golf/data/스윙분석/Golf_Swing_Slow_2person.mp4'
video_path = 'E:/Golf/data/프로스윙2/전인지_드라이버_정면_슬로모션.mp4'  # 120-1200
#video_path = 'E:/Golf/data/프로스윙2/전인지_드라이버_측면.mp4' #120-1200
vid = cv2.VideoCapture(video_path)
stick = 'E:/Golf/data/stick.png'

'''
out_vid = cv2.VideoWriter('E:/Golf/data/prev-next-masking.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          vid.get(cv2.CAP_PROP_FPS),
                          (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
'''

start = 120
last = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
sampling = 4
vid.set(cv2.CAP_PROP_POS_FRAMES, start)
_, start_frame = vid.read()

vid.set(cv2.CAP_PROP_POS_FRAMES, last)
_, last_frame = vid.read()

print('interval {} - {} frames'.format(start, last))
vid.set(cv2.CAP_PROP_POS_FRAMES, start)

cnt = start
prev = curr = next = None
cv2.namedWindow('curr', cv2.WINDOW_NORMAL)
cv2.namedWindow('prev', cv2.WINDOW_NORMAL)
cv2.namedWindow('next', cv2.WINDOW_NORMAL)
# cv2.namedWindow('prev_dif',cv2.WINDOW_NORMAL)
# cv2.namedWindow('next_dif',cv2.WINDOW_NORMAL)
# cv2.namedWindow('dif',cv2.WINDOW_NORMAL)
# cv2.namedWindow('thr',cv2.WINDOW_NORMAL)
cv2.namedWindow('prev_thr', cv2.WINDOW_NORMAL)
cv2.namedWindow('next_thr', cv2.WINDOW_NORMAL)
cv2.namedWindow('stick_prev', cv2.WINDOW_NORMAL)
cv2.namedWindow('stick_curr', cv2.WINDOW_NORMAL)
cv2.namedWindow('stick_next', cv2.WINDOW_NORMAL)
cv2.namedWindow('act_m', cv2.WINDOW_NORMAL)
cv2.namedWindow('act', cv2.WINDOW_NORMAL)
cv2.namedWindow('act2', cv2.WINDOW_NORMAL)
cv2.namedWindow('act3', cv2.WINDOW_NORMAL)
cv2.namedWindow('motion', cv2.WINDOW_NORMAL)
cv2.namedWindow('cntr', cv2.WINDOW_NORMAL)
cv2.namedWindow('visual', cv2.WINDOW_NORMAL)

while vid.isOpened():
    ret, frame = vid.read()
    if not ret: break
    if cnt == start:
        prev = frame.copy()
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        motion = cv2.cvtColor(np.zeros_like(frame), cv2.COLOR_BGR2BGRA)
        motion[:, :, :] = 0
        cnt += 1
        continue
    visual = frame.copy()
    if cnt % sampling == 0:
        if curr is None:
            curr = frame.copy()
            gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            prev_dif = cv2.absdiff(gray, prev_gray)
            cnt += 1
            continue

        next = frame.copy()
        next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.GaussianBlur(next_gray, (5, 5), 0)
        next_dif = cv2.absdiff(gray, next_gray)

        prev_next_dif = cv2.absdiff(prev_gray, next_gray)

        cv2.imshow('prev', prev)
        cv2.imshow('curr', curr)
        cv2.imshow('next', next)

        # dif=cv2.addWeighted(next_dif,0.5,prev_dif,0.5,0)
        _, next_thr = cv2.threshold(next_dif, 5, 255, cv2.THRESH_BINARY)
        _, prev_thr = cv2.threshold(prev_dif, 5, 255, cv2.THRESH_BINARY)
        _, prev_next_thr = cv2.threshold(prev_next_dif, 5, 255, cv2.THRESH_BINARY)
        # thr=cv2.bitwise_or(prev_thr,next_thr)
        # _, thr = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)

        # print(np.max(np.mean([next_dif,prev_dif])))

        prev_contours, hierachy = cv2.findContours(prev_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prev_action_mask = np.zeros_like(gray)
        for cntr in prev_contours:
            # if cv2.arcLength(cntr, True) >= 1000:
            # if cv2.contourArea(cntr)<3000:
            cv2.drawContours(image=visual, contours=[cntr], contourIdx=-1, color=255, thickness=-1)
            cv2.drawContours(image=prev_action_mask, contours=[cntr], contourIdx=-1, color=255, thickness=-1)
        next_contours, hierachy = cv2.findContours(next_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        next_action_mask = np.zeros_like(gray)
        for cntr in next_contours:
            # if cv2.contourArea(cntr) < 3000:
            cv2.drawContours(image=visual, contours=[cntr], contourIdx=-1, color=(0, 0, 255), thickness=-1)
            cv2.drawContours(image=next_action_mask, contours=[cntr], contourIdx=-1, color=255, thickness=-1)
        prev_next_contours, hierachy = cv2.findContours(prev_next_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prev_next_action_mask = np.zeros_like(gray)
        for cntr in prev_next_contours:
            # if cv2.contourArea(cntr) < 3000:
            cv2.drawContours(image=visual, contours=[cntr], contourIdx=-1, color=(0, 255, 0), thickness=-1)
            cv2.drawContours(image=prev_next_action_mask, contours=[cntr], contourIdx=-1, color=255, thickness=-1)

        cv2.imshow('cntr', visual)
        cv2.imshow('prev_action_mask', prev_action_mask)
        cv2.imshow('next_action_mask', next_action_mask)
        cv2.imshow('prev_next_action_mask', prev_next_action_mask)


        # curr 에서 prev, next와 겹치지 않는 부분만 남김
        aa=cv2.bitwise_and(prev_action_mask, cv2.bitwise_not(prev_next_action_mask))
        bb = cv2.bitwise_and(next_action_mask, cv2.bitwise_not(prev_next_action_mask))
        cv2.imshow('aa',aa)
        cv2.imshow('bb', bb)
        cv2.imshow('cc', cv2.absdiff(aa,bb))
        cv2.imshow('dd', cv2.bitwise_and(aa, bb))

        #cv2.imshow('vvv', cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))

        action_mask = bb#cv2.bitwise_and(aa, bb)
        # motion accumulate - simple
        motion[:, :, 3] = cv2.bitwise_or(motion[:, :, 3], action_mask)
        motion[:, :, :3][action_mask != 0] = curr[action_mask != 0]

        prev = curr
        prev_gray = gray
        curr = next
        gray = next_gray
        prev_dif = next_dif

    # visual[:, :, :3][motion[:, :, 3] == 0] = frame[:, :, :3][motion[:, :, 3] == 0]
    visual[:, :, :3][motion[:, :, 3] != 0] = motion[:, :, :3][motion[:, :, 3] != 0]

    cv2.imshow('visual', visual)
    cv2.imshow('motion', motion)
    # out_vid.write(visual)

    cnt += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

motion[:, :, :3][motion[:, :, 3] == 0] = curr[motion[:, :, 3] == 0]
rr = cv2.bitwise_and(curr, motion[:, :, :3], mask=motion[:, :, 3])
bb = cv2.absdiff(motion[:, :, :3], curr)

# cv2.imshow('rr',rr)
# cv2.imshow('bb',bb)
# cv2.waitKey()

while vid.isOpened():
    ret, frame = vid.read()
    if not ret: break
    visual = frame.copy()

    # visual[:, :, :3][motion[:, :, 3] == 0] = frame[:, :, :3][motion[:, :, 3] == 0]
    visual[:, :, :3][motion[:, :, 3] != 0] = motion[:, :, :3][motion[:, :, 3] != 0]
    cv2.imshow('visual', visual)
    # out_vid.write(visual)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# motion[:, :, :3][motion[:, :, 3] == 0] = last_frame[motion[:, :, 3] == 0]
# out = cv2.cvtColor(motion, cv2.COLOR_BGRA2BGR)
# out_vid.write(out)
# cv2.imshow('visual', visual)
cv2.waitKey(0)
# out = cv2.addWeighted(start_frame, 0.5, out, 0.5, 0)
# out_vid.write(out)
# out = cv2.addWeighted(last_frame, 0.5, motion, 0.8, 0)

# cv2.imshow('out', out)
# cv2.waitKey(0)

cv2.destroyAllWindows()
