'''
stereo vision task: given the left and right image find
assumptions
baseline = 100mm
focal length = 2.8mm (for both cameras)
'''
#TODO the fundmental matrix
#import the modules
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

left = cv.imread('000023L.png',0)   # left image
right = cv.imread('000023R.png',0)   # right image
sift = cv.xfeatures2d.SIFT_create() #Scale Invariant Feature Transform

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(left,None)
kp2, des2 = sift.detectAndCompute(right,None)


# FLANN find match between two images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
'''
This matcher trains cv::flann::Index on a train descriptor collection and calls
its nearest search methods to find the best matches.
So, this matcher may be faster when matching a large train collection than the brute force matcher. FlannBasedMatcher does not support masking permissible matches of descriptor sets
because flann::Index does not support this.
'''
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS) #fundmental matrix
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#draw epilines of the points
#img1 - image on which we draw the epilines for the points in img2lines - corresponding epilines
def drawlines(left,right,lines,pts1,pts2):
    r,c = left.shape
    left = cv.cvtColor(left,cv.COLOR_GRAY2BGR)
    right = cv.cvtColor(right,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist()) #choose a random color
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        left = cv.line(left, (x0,y0), (x1,y1), color,1)
        left = cv.circle(left,tuple(pt1),5,color,-1)
        right = cv.circle(right,tuple(pt2),5,color,-1)
    return left,right

# drawing its lines on left image
'''
For points in an image of a stereo pair,
computes the corresponding epilines in the other image.
'''
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F) #the points, in which image, F = fundmentl matrix
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(left,right,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(left,right,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
