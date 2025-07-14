import cv2
import json

img = cv2.imread('frame_proc.png')
pts_img = []

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts_img) < 4:
        pts_img.append([x, y])
        print(len(pts_img), [x, y])

cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_cb)

while len(pts_img) < 4:
    cv2.imshow('img', img)
    cv2.waitKey(20)

cv2.destroyAllWindows()
json.dump(pts_img, open('pts_img.json', 'w'))

