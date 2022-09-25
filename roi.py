import cv2
import matplotlib.pyplot as plt

# image_path
img_path = ''

# read image
img_raw = cv2.imread(img_path)

# select ROI function
roi = cv2.selectROI(img_raw)
# roi = (482, 379, 534, 172)

# print rectangle points of selected roi
print(roi)

# Crop selected roi from raw image
roi_cropped = img_raw[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
# offset y, offset x, size y, size x

# show
plt_roi = cv2.cvtColor(roi_cropped, cv2.COLOR_RGB2BGR)
plt.imshow(plt_roi)
plt.show()

# cv2.imwrite("crop_04-a06a1.jpeg", roi_cropped)
