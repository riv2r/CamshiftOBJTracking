import cv2
import numpy as np

xs, ys, ws, hs = 0, 0, 0, 0
xo, yo = 0, 0
selectObject = False
trackObject = 0

trackPath = []


def onMouse(event, x, y, flags, prams):
    global xs, ys, ws, hs, selectObject, xo, yo, trackObject
    if selectObject == True:
        xs = min(x, xo)
        ys = min(y, yo)
        ws = abs(x - xo)
        hs = abs(y - yo)
    if event == cv2.EVENT_LBUTTONDOWN:
        xo, yo = x, y
        xs, ys, ws, hs = x, y, 0, 0
        selectObject = True
    elif event == cv2.EVENT_LBUTTONUP:
        selectObject = False
        trackObject = -1


cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.namedWindow('CamshiftOBJTracking')
cv2.setMouseCallback('CamshiftOBJTracking', onMouse)
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while (True):
    ret, frame = cap.read()
    if trackObject != 0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array((0., 30., 10.)), np.array((180., 256., 255.)))
        if trackObject == -1:
            track_window = (xs, ys, ws, hs)
            mask_roi = mask[ys:ys + hs, xs:xs + ws]
            hsv_roi = hsv[ys:ys + hs, xs:xs + ws]
            roi_hist = cv2.calcHist([hsv_roi], [0], mask_roi, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            trackObject = 1
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        dst &= mask
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        '''
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        '''
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        trackPath.append([x + w // 2, y + h // 2])

        for i in range(1, len(trackPath)):
            cv2.line(frame, (trackPath[i - 1][0], trackPath[i - 1][1]), (trackPath[i][0], trackPath[i][1]), (0, 255, 0),
                     2)

    if selectObject and ws > 0 and hs > 0:
        cv2.bitwise_not(frame[ys:ys + hs, xs:xs + ws], frame[ys:ys + hs, xs:xs + ws])

    cv2.imshow('CamshiftOBJTracking', frame)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
cap.release()
