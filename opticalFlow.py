#imports
import numpy as np
import cv2


# need contrasting background in gray scale

if __name__ == '__main__':
	cap = cv2.VideoCapture('assemble2.mov')
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# paramaters for Shi Tomasi corner detection
	feature_params = dict( maxCorners = 100,
	                       qualityLevel = 0.3,
	                       minDistance = 7,
	                       blockSize = 7 )

	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
	                  maxLevel = 2,
	                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# Create some random colors
	color = np.random.randint(0,255,(100,3))

	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	origImg = old_frame
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	print("P0***********************************")
	print(p0)


	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_frame)
	arrMask = np.zeros_like(old_frame)



	k = 0;

	while(k <= length):
	    ret,frame = cap.read()
	    # check if frame is empty
	    if np.shape(frame) == ():
	        break
	    else:
	    	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

	    # calculate optical flow
	    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	    # Select good points
	    good_new = p1[st==1]
	    good_old = p0[st==1]


	    # save first and last frames for arrows
	    if(k == 0):
	    	firstFrame = good_old
	    else:
	    	lastFrame = good_new

	    # draw the tracks
	    for i,(new,old) in enumerate(zip(good_new,good_old)):
	        a,b = new.ravel()
	        c,d = old.ravel()
	        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
	        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
	        
	    img = cv2.add(frame,mask)

	    cv2.imshow('frame',img)
	    if cv2.waitKey(10) & 0xFF == ord('q'):
	        break

	    # Now update the previous frame and previous points
	    old_gray = frame_gray.copy()
	    p0 = good_new.reshape(-1,1,2)

	    k += 1


	cv2.waitKey(0)


	

	# add optical flow arrows to first frame
	for i,(new,old) in enumerate(zip(firstFrame,lastFrame)):
		a,b = new.ravel()
		c,d = old.ravel()

		arrMask = cv2.arrowedLine(arrMask, (a,b), (c,d), color=(0, 0, 255), thickness=2, tipLength=.2)
		arrFrame = cv2.circle(old_frame,(a,b),5,(0, 0, 255),-1)

	newImage = cv2.add(arrFrame,arrMask)

	cv2.imshow('frame',newImage)
	cv2.waitKey(0)

	cv2.destroyAllWindows()
	cap.release()











