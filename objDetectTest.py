import cv2 
import numpy as np 
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('/Users/coreyhannum/Desktop/Thesis/assemble2.mov')
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
					maxLevel = 2,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#colors for optical flow lines
color = np.random.randint(0,255,(100,3))

# convert frame to grayscale
ret, old_frame = cap.read()
origImg = old_frame
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# generate masks
mask = np.zeros_like(old_frame)
arrMask = np.zeros_like(old_frame)

# initiate MIL tracker
tracker = cv2.TrackerMIL_create()

# initiate tracker boxes and optical flow points, then calculate center of boxes
boxes = []
ofp0 = []
box1 = cv2.selectROI('MultiTracker', old_frame, False)
boxes.append(box1)

p0 = (int(box1[0]), int(box1[1]))
p1 = (int(box1[0] + box1[2]), int(box1[1] + box1[3]))
ofp0.append(np.float32((int((p0[0]+p1[0])/2), int((p0[1]+p1[1])/2))))

box2 = cv2.selectROI('MultiTracker', old_frame, False)
boxes.append(box2)

p0 = (int(box2[0]), int(box2[1]))
p1 = (int(box2[0] + box2[2]), int(box2[1] + box2[3]))
ofp0.append(np.float32((int((p0[0]+p1[0])/2), int((p0[1]+p1[1])/2))))

# convert optical flow points to np array
ofp0 = np.array(ofp0)


print(ofp0)

# initiate multi tracker
multiTracker = cv2.MultiTracker_create()
for box in boxes:
	multiTracker.add(cv2.TrackerMIL_create(),old_frame, box)


# i = 0

# calculate optical flow and track selected boxes
while(True):
	ret, new_frame = cap.read()
	if np.shape(new_frame) == ():
		break
	else:
		frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  

	ret, boxes = multiTracker.update(new_frame)

	# track each box selected
	for boxUpdate in boxes:
		p0 = (int(boxUpdate[0]), int(boxUpdate[1]))
		p1 = (int(boxUpdate[0] + boxUpdate[2]), int(boxUpdate[1] + boxUpdate[3]))
		circleCent = (int((p0[0]+p1[0])/2), int((p0[1]+p1[1])/2))
		cv2.circle(new_frame, circleCent, 100, (0,0,255), 2, 1)
		cv2.circle(new_frame, circleCent, 1, (0,0,255), 2, 1)

		# calc optical flow
		ofp1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, ofp0, None, **lk_params)
		good_new = ofp1
		good_old = ofp0

		# draw optical flow path
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
			new_frame = cv2.circle(new_frame,(a,b),5,color[i].tolist(),-1)
			print(good_new)

		img = cv2.add(new_frame,mask)
		cv2.imshow('frame',img)

	old_gray = frame_gray.copy()
	ofp0 = good_new.reshape(-1,1,2)

	# i += 1

	j = cv2.waitKey(1) & 0xff
	if j == 27 : break

cv2.waitKey(0)