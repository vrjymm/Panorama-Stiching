import numpy as np
import cv2
import sys

# Run: python main.py path/to/right/image path/to/left/image

""" Step 1: Image Capture and Processing """

img1 = cv2.imread(sys.argv[1])

img2 = cv2.imread(sys.argv[2])

scale_down = 0.2
img1 = cv2.resize(img1, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
img2 = cv2.resize(img2, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)

print("images scaled down -")
print(f"image 1 dims: {img1.shape}")
print(f"image 2 dims: {img2.shape}")

cv2.imshow('image1',img1)
cv2.imshow('image2',img2)


img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

""" Step 2: SIFT image key points and descriptors detection using opencv """
sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1_gray,None)
kp2, des2 = sift.detectAndCompute(img2_gray,None)

n_kp1 = len(kp1) # no. of SIFT key points in the first image
n_kp2 = len(kp2) # no. of SIFT key points in the second image

print(f"keypoint eg: {kp1[0].pt}")
img1_kp=cv2.drawKeypoints(img1_gray,kp1,img1_gray)
img2_kp=cv2.drawKeypoints(img2_gray,kp2,img2_gray)

cv2.imwrite("images/image1_keypoints.png", img1_kp)

cv2.imwrite("images/image2_keypoints.png", img2_kp)

# cv2.imshow('image1_keypoints',img1_kp)


# print(f"kp1: {len(kp1)}")
# print(f"kp2: {len(kp2)}")

# each descriptor is a (kp, 128) size numpy array
# print(f"des1: {des1.shape}")
# print(f"des2: {des2.shape}")

""" Step 3: Feature Mapping """

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Feature Mapping",img3)

cv2.imwrite("images/feature_mapping.png", img3)

""" Step 4: Homography Estimation using RANSAC """
obj = np.empty((len(good),2), dtype=np.float32)
scene = np.empty((len(good),2), dtype=np.float32)
for i in range(len(good)):
    
    #-- Get the keypoints from the good matches
    obj[i,0] = kp1[good[i][0].queryIdx].pt[0]
    obj[i,1] = kp1[good[i][0].queryIdx].pt[1]
    scene[i,0] = kp2[good[i][0].trainIdx].pt[0]
    scene[i,1] = kp2[good[i][0].trainIdx].pt[1]

H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)


""" Step 5: Image Stitching """

# warping img1 into the plane of img2 using the homography matrix
dst = cv2.warpPerspective(img1,H,(img1.shape[1] + img2.shape[1], img2.shape[0]))
dst[0:img2.shape[0],0:img2.shape[1]] = img2
cv2.imwrite("images/image_stitched.png", dst)
cv2.imshow("original_image_stitched.jpg", dst)


cv2.waitKey(0)
cv2.destroyAllWindows()