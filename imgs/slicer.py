from PIL import Image
import cv2

test_image = "3x3_sample.png"

im =  cv2.imread(test_image)

imgheight=im.shape[0]
imgwidth=im.shape[1]

y1 = 0
M = imgheight//3
N = imgwidth//3

for x in range(0, imgheight, N):
    for y in range(0, imgwidth, M):
        y1 = y + M
        x1 = x + N
        tiles = im[x:x+N,y:y+M,]
        cv2.rectangle(im, (x, y), (x1, y1), (0, 0, 0))
        cv2.imwrite(str(x) + '_' + str(y)+".png",tiles)

cv2.imwrite("grid_test.png",im)