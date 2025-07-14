import numpy as np, cv2, json

pts_img   = np.array(json.load(open('pts_img.json')), dtype=np.float32)
pts_world = np.array([[0,0],
                      [23.77,0],
                      [23.77,8.23], [0,8.23]], dtype=np.float32)

H, _ = cv2.findHomography(pts_img, pts_world, method=cv2.RANSAC)
np.save('H.npy', H)
print('✓ homography saved')

