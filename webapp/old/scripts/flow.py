import numpy as np
import cv2 as cv
import matplotlib.colors as mcolors
def h2r(h): return tuple(int(h[1:][i:i+2], 16) for i in (0, 2, 4))

class OpticalFlow():
    def __init__(self):
        self.feature_params = dict( maxCorners = 100,
		               qualityLevel = 0.3,
		               minDistance = 7,
		               blockSize = 7 )

        self.lk_params = dict( winSize  = (15, 15),
		          maxLevel = 2,
		          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

        self.color = [h2r(color) for color in mcolors.CSS4_COLORS.values()]# 
        self.color = np.random.randint(0, 255, (100, 3))
        self.old_gray = 0
        self.p0 = 0
        
        self.term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
        
    def get_mean_shift(self,p0,frame,hsv):
        cx , cy = p0[0].ravel()
        x, y, w, h =  int(cx), int(cy), 100, 50# simply hardcoded the values
        track_window = (x, y, w, h)
        print(track_window)
        roi = frame[y:y+h, x:x+w]
        hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

        dst = cv.calcBackProject([hsv],[0], roi_hist,[0,180],1)
        ret, track_window = cv.meanShift(dst, track_window, self.term_crit)
        x,y,w,h = track_window
        mean_shift = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        dst =  cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        return dst#mean_shift
                
    def get_flow(self,frame):
        if type(self.old_gray) == int:
           self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
           self.p0 = cv.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
           return frame, frame, 0, 0
        
        old_gray = self.old_gray; p0 = self.p0 
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        
        mask = np.zeros_like(frame)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0, None, **self.lk_params)     
        
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        if len(st) <=5 :
            good_new = good_old = p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **self.feature_params)
        
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)),  self.color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
	    
        flow_1 = cv.add(frame, mask)
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        print(len(st))
        
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        flow_2 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)      
        
        
        #mean_shift = self.get_mean_shift(good_new,frame,hsv)
        
        
        return flow_1 ,flow_2,str(len(st)),str(hsv.mean())
