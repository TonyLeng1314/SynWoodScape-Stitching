import cv2
import numpy as np
from PGImg import Undistorter,TransMatrixDeriver,CoCalibrater,get_point_list

front = cv2.imread('./imgs/front.png')
left = cv2.imread('./imgs/left.png')
rear = cv2.imread('./imgs/rear.png')
right = cv2.imread('./imgs/right.png')

vision_name_list = ['fornt','left','rear','right']

img_before_list = [front,left,rear,right]
img_list = []

ud = Undistorter('./MVL.json')
ud.get_maps_from_json('./maps.json')

for img in img_before_list:
    img_processed = ud.undistort(img)
    img_list.append(img_processed)
    
    
tmd_list = []
plist = get_point_list((100,100),(100,100),(100,200),(10,10))
for i,p in enumerate(plist[0]):
    tmd = TransMatrixDeriver(vision_name_list[i],p,img_list[i])
    tmd_list.append(tmd)






coclber = CoCalibrater(img_list,pointlists=plist,tmdlist=tmd_list)

print(plist)

coclber.start_calibration()



# cv2.waitKey(0)
