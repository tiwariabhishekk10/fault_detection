import imutils
import cv2

from skimage.measure import compare_ssim #ssim: structural similarity index

image_good=cv2.imread('D:/image/good_image.png')
image_good

image_bad=cv2.imread('D:/image/bad_image.png')
image_bad

# convert the images to grayscale
gray_good=cv2.cvtColor(image_good, cv2.COLOR_BGR2GRAY)
gray_fault=cv2.cvtColor(image_bad, cv2.COLOR_BGR2GRAY)

# compute the difference between two images
(score, diff)=compare_ssim(gray_good, gray_fault, full=True)
diff=(diff*224).astype("uint8")
print("SSIM: {}".format(score))

#obtain the regions of the two input images that differ
thresh=cv2.threshold(diff, 0, 256,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(image_good, (x, y), (x + w, y + h), (0, 0, 224), 2)
	cv2.rectangle(image_bad, (x, y), (x + w, y + h), (0, 0, 224), 2)
# show the output images
cv2.imshow("Original", image_good)
cv2.imshow("Modified", image_bad)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)