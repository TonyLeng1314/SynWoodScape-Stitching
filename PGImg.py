from turtle import onclick
import cv2
import cv2.img_hash
from sympy import true
import numpy as np
import json

'''
  "intrinsic": {
    "aspect_ratio": 1.00021,
    "cx_offset": 0.329055,
    "cy_offset": -3.5506,
    "height": 966.0,
    "k1": 341.725,
    "k2": -26.4448,
    "k3": 32.7864,
    "k4": 0.50499,
    "model": "radial_poly",
    "poly_order": 4,
    "width": 1280.0
'''

'''
front = cv2.imread("./imgs/left.png")

cv2.imshow('hi',front)
cv2.waitKey(0)

K = np.array([[-0.005],[-0.005],[-0.0012],[-0.001]])

# K = np.array([0.,0,0,0])
w = 1280
fov = 190
f = (w/(2*np.tan(np.pi*fov/360)))
print(f)
# for i in range(10000,20000,10):
f = 360
Intrinsic = np.array([
    [f,0,640],
    [0,f,480.],
    [0,0,1]
])

Knew = np.array([
    [1,0,640.],
    [0,1,480.],
    [0,0,1]
])

map1,map2 = cv2.initUndistortRectifyMap(Intrinsic,K,np.eye(3),Intrinsic,(1280,960),cv2.CV_16SC2)
front_undistort = cv2.remap(front,map1,map2,cv2.INTER_LINEAR)

cv2.imshow('hi',front_undistort)
key = cv2.waitKey(0)
if key == 13:
    # break
    pass
'''



class Undistorter():
    def __init__(self,json_file) -> None:
        with open(json_file,'r') as f:
            data = json.load(f)
        intrinsic_data = data["intrinsic"]
        self.width = int(intrinsic_data['width'])
        self.height = int(intrinsic_data['height'])
        self.shape = (self.height,self.width)
        self.D = np.array([
            [intrinsic_data['k1']],
            [intrinsic_data['k2']],
            [intrinsic_data['k3']],
            [intrinsic_data['k4']],
        ])
        
        self.cx = intrinsic_data['cx_offset']
        self.cy = intrinsic_data['cy_offset']
        self.aspect_ratio = intrinsic_data['aspect_ratio']
        self.map1 = np.empty(self.shape)
        self.map2 = np.empty(self.shape)

    def get_maps(self,Z:float,xrange:tuple,yrange:tuple):
        map1 = np.zeros((self.height,self.width),dtype=np.float32)
        map2 = np.zeros((self.height,self.width),dtype=np.float32)
        
        for X in range(self.width):
            for Y in range(self.height):
                x = X*(xrange[1]-xrange[0])/self.width + xrange[0] 
                y = Y*(yrange[1]-yrange[0])/self.height + yrange[0] 
                chi = np.sqrt(x**2+y**2)
                theta = np.arctan2(chi,Z)
                rho = np.array([theta,theta**2,theta**3,theta**4],dtype=np.float32)@self.D
                u = rho.item()*x/chi if chi!=0 else 0
                v = rho.item()*y/chi if chi!=0 else 0
                map1[Y,X] = u+self.cx +self.width/2 - 0.5
                map2[Y,X] = v*self.aspect_ratio + self.cy + self.height/2 +0.5
                self.map1 = map1
                self.map2 = map2
                
    def undistort(self,img):
        return cv2.remap(img,self.map1,self.map2,interpolation=cv2.INTER_LINEAR)
    
    def save_maps(self):
        map1_list = self.map1.tolist()
        map2_list = self.map2.tolist()
        maps_dict = {"map1":map1_list,"map2":map2_list}
        with open('maps.json','w') as json_file:
            json.dump(maps_dict,json_file)
        print('saved maps successfully!')
        
    def get_maps_from_json(self,json_file):
        with open(json_file,'r') as f:
            data = json.load(f)
        self.map1 = np.array(data['map1'],dtype=np.float32)
        self.map2 = np.array(data['map2'],dtype=np.float32)
        
        
        
class TransMatrixDeriver():
    def __init__(self,vision_name,point_list:np.ndarray,img:np.ndarray) -> None:
        self.point_list = point_list
        self.vision_name = vision_name
        self.click_list = []
        self.ml_down = False
        self.nearest_click_id = -1
        self.img = img
        self.img_warpped = img
        
        
    def get_distance(self,p1,p2):
        return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)    
    
    def get_nearest_id(self,mouse_point):
        min = np.inf
        id = 0
        for i,point in enumerate(self.click_list):
            distance = self.get_distance(mouse_point,point)
            if distance<min:
                min = distance
                id = i
        return id
            
    def display_update(self):
        img_show = self.img.copy()
        img_warpped_show = self.img_warpped.copy()
        for point in self.point_list:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(img_warpped_show,(x,y),1,(255,0,0),1)
            cv2.circle(img_warpped_show,(x,y),5,(5,0,255),2)
        for point in self.click_list:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(img_show,(x,y),1,(255,0,0),1)
            cv2.circle(img_show,(x,y),5,(5,0,255),2)
        cv2.imshow(self.vision_name+'_before',img_show)
        cv2.imshow(self.vision_name+'_after',img_warpped_show)
        
    def display_co_update(self):
        img_show = self.img.copy()
        for point in self.click_list:
            x = int(point[0])
            y = int(point[1])
            cv2.circle(img_show,(x,y),1,(255,0,0),1)
            cv2.circle(img_show,(x,y),5,(5,0,255),2)
        cv2.imshow(self.vision_name+'_before',img_show)
    
    def on_click(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ml_down = True
            if len(self.click_list) < 4:
                self.click_list.append([x,y])
                self.display_update()
            else:
                self.nearest_click_id = self.get_nearest_id((x,y))
            print(len(self.click_list))
            
        if event == cv2.EVENT_LBUTTONUP:
            self.ml_down = False
            
        if event == cv2.EVENT_MOUSEMOVE:
            if self.ml_down == True and len(self.click_list) >=4 :
                self.click_list[self.nearest_click_id] = [x,y]
                self.display_update()
                print('moved!')
        
    def on_co_click(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ml_down = True
            if len(self.click_list) < 4:
                self.click_list.append([x,y])
                self.display_co_update()
                
            else:
                self.nearest_click_id = self.get_nearest_id((x,y))
            print(len(self.click_list))
            
        if event == cv2.EVENT_LBUTTONUP:
            self.ml_down = False
            
        if event == cv2.EVENT_MOUSEMOVE:
            if self.ml_down == True and len(self.click_list) >=4 :
                self.click_list[self.nearest_click_id] = [x,y]
                self.display_co_update()
                print('moved!')

    def start_calibration(self,img:np.ndarray,final_shape:tuple):
        cv2.imshow(self.vision_name+"_before",self.img)
        cv2.namedWindow(self.vision_name+"_after")
        cv2.setMouseCallback(self.vision_name+'_before',self.on_click)
        
        while(True):
            # self.display_update()
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
            
            if len(self.click_list) >= 4 :
                # print("go!")
                H,_ = cv2.findHomography(np.array(self.click_list,dtype=np.float32),self.point_list)
                self.img_warpped = cv2.warpPerspective(self.img,H,final_shape)
                self.display_update()

    def do_co_calibration(self):
        cv2.namedWindow(self.vision_name+'_before',cv2.WINDOW_FREERATIO)
        cv2.setMouseCallback(self.vision_name+'_before',self.on_co_click)
    
    def get_H(self):
        if len(self.click_list) < 4:
            return np.eye(3,dtype=np.float32)
        elif len(self.click_list) == 4:
            H,_ = cv2.findHomography(np.array(self.click_list,dtype=np.float32),self.point_list)
            return H
            



class CoCalibrater():
    def __init__(self,imglist:list[np.ndarray],pointlists:tuple=(),tmdlist:list[TransMatrixDeriver]=[]) -> None:
        self.img_list = imglist
        self.point_lists = pointlists[0]
        self.final_shape = (pointlists[2],pointlists[1])
        self.tmdlist = tmdlist
        self.final_img = np.zeros((self.final_shape[0],self.final_shape[1],3),dtype = np.uint8)
        self.H_list = []
        for i in range(4):
            H = np.eye(3,dtype = np.float32)
            self.H_list.append(H)
            
        self.load_mode = False
        
    def get_pos_from_point_list(self,i):
        y0,y1,x0,x1 = 0,1,0,1
        if i == 0 :
            y0 = int(0)
            y1 = int(self.point_lists[i][2][1])
            x0 = int(0)
            x1 = int(self.final_shape[1])
        elif i == 1 :
            y0 = int(self.point_lists[i][1][1])
            # y0 = 100
            y1 = int(self.point_lists[i][2][1])
            x0 = int(0)
            x1 = int(self.point_lists[i][2][0])
        elif i == 2 :
            y0 = int(self.point_lists[i][2][1])
            y1 = int(self.final_shape[0])
            x0 = int(0)
            x1 = int(self.final_shape[1])
        elif i == 3 :
            y0 = int(self.point_lists[i][2][1])
            y1 = int(self.point_lists[i][3][1])
            x0 = int(self.point_lists[i][2][0])
            x1 = int(self.final_shape[1])
        return y0,y1,x0,x1
    
    def get_overlap_pose_from_point_list(self,i):
        y0,y1,x0,x1 = 0,1,0,1
        if i == 0 :
            y0 = int(0)
            y1 = int(self.point_lists[i][2][1])
            x0 = int(0)
            x1 = int(self.point_lists[i][0][0])
        elif i == 1 :
            y0 = int(self.point_lists[i][0][1])
            y1 = int(self.final_shape[0])
            x0 = int(0)
            x1 = int(self.point_lists[i][2][0])
        elif i == 2 :
            y0 = int(self.point_lists[i][2][1])
            y1 = int(self.final_shape[0])
            x0 = int(self.point_lists[i][0][0])
            x1 = int(self.final_shape[1])
        elif i == 3 :
            y0 = int(0)
            y1 = int(self.point_lists[i][0][1])
            x0 = int(self.point_lists[i][2][0])
            x1 = int(self.final_shape[1])
        return y0,y1,x0,x1
    
        
    def save_H_list(self):
        Hs = []
        for H in self.H_list:
            Hi = H.tolist()
            Hs.append(Hi)

        Hdick = {'H_list.json':Hs}
        with open('H_list','w') as f:
            json.dump(Hdick,f)
            print("H_list saved successfully!")
            
            
    def load_H_lsit(self):
        with open('H_list.json','r')as f:
            Hdick = json.load(f)
        self.H_list = np.array(Hdick['H_list'],dtype=np.float32)
        
        
    def get_mask(self):
        y0,y1,x0,x1 = self.get_overlap_pose_from_point_list(0)
        sizex = x1
        sizey = y1
        self.mask = np.zeros((y1,x1,4),dtype = np.uint8)
        mask1_list = []
        for i,tmd in enumerate(self.tmdlist):
            ret,mask1 = cv2.threshold(tmd.img_warpped,5,255,cv2.THRESH_BINARY)
            mask1 = cv2.cvtColor(mask1,cv2.COLOR_BGR2GRAY)
            mask1_list.append(mask1)

        mask4_list = []
        mask5_list = []
        for i in range(4):
            if i == 3:
                j = 0
            else:
                j = i+1
            y0,y1,x0,x1 = self.get_overlap_pose_from_point_list(i)
            mask2 = mask1_list[i][y0:y1,x0:x1]
            mask3 = mask1_list[j][y0:y1,x0:x1]
            mask_overlap = cv2.bitwise_and(mask2,mask3)
            mask4 = cv2.bitwise_xor(mask2,mask_overlap)
            mask5 = cv2.bitwise_xor(mask3,mask_overlap)
            mask4_list.append(mask4)
            mask5_list.append(mask5)
            
        # for mask1,mask2 in zip(mask4_list,mask5_list):
        #     cv2.imshow('4',mask1)
        #     cv2.imshow('5',mask2)
        #     cv2.waitKey()
        
        for i in range(4):
            mask_left = mask4_list[i]
            mask_right = mask5_list[i]
            contours1,_ = cv2.findContours(mask_left,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contours2,_ = cv2.findContours(mask_right,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            mask_i = np.zeros((sizey,sizex),dtype=np.uint8)
            for y in range(sizey):
                for x in range(sizex):
                    d1 = cv2.pointPolygonTest(contours1[0],(x,y),True)
                    d2 = cv2.pointPolygonTest(contours2[0],(x,y),True)
                    value = d2**2/(d1**2+d2**2+0.000001)
                    mask_i[y,x] = value*255
            mask_i = cv2.bitwise_or(mask_i,mask_left)
            mask_i = cv2.bitwise_and(cv2.bitwise_not(mask_right),mask_i)
                    
            self.mask[:,:,i] = mask_i
            cv2.imshow('hi',mask_i)
            cv2.waitKey()
            cv2.imwrite('./mask.png',self.mask)
            
    def get_weight_from_mask(self):
            self.weight = self.mask.astype(np.float32)
            self.weight = self.weight/255
            
    def get_mask_from_png(self,path:str):
        mask = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        print(mask.shape)
        self.mask = mask
        
    
    def polish(self):
        self.get_weight_from_mask()
        _,sizey,_,sizex= self.get_overlap_pose_from_point_list(0)
        for i in range(4):
            if i == 3 :
                j = 0
            else:
                j = i+1
            y0,y1,x0,x1 = self.get_overlap_pose_from_point_list(i)
            weight_now = np.repeat(np.reshape(self.weight[:,:,i],(sizey,sizex,1)),3,2)
            img_left = self.tmdlist[i].img_warpped[y0:y1,x0:x1].astype(np.float32)
            img_right = self.tmdlist[j].img_warpped[y0:y1,x0:x1].astype(np.float32)
            result = cv2.multiply(img_left,weight_now).astype(np.uint8)+cv2.multiply(img_right,1-weight_now).astype(np.uint8)
            self.final_img[y0:y1,x0:x1] = result
     
    def start_calibration(self ):
        while(True):
            key = cv2.waitKey(10)
            
            self.final_img = np.zeros((self.final_shape[0],self.final_shape[1],3),dtype = np.uint8)
            for i,tmd in enumerate(self.tmdlist):
                tmd.do_co_calibration()
                if self.load_mode == False:
                    self.H_list[i] = tmd.get_H()
                img_warpped = cv2.warpPerspective(tmd.img,self.H_list[i],(self.final_shape[1],self.final_shape[0]))
                tmd.img_warpped = img_warpped
                y0,y1,x0,x1 = self.get_pos_from_point_list(i)
                
                self.final_img[y0:y1,x0:x1] += img_warpped[y0:y1,x0:x1] # type: ignore
                cv2.imshow('final',self.final_img)
                
                
            if key == ord('q'):
                break
            if key == ord('x'):
                print(self.H_list)
                self.save_H_list()
                
            if key == ord('l'):
                self.load_H_lsit()
                print("H_list loaded!")
                self.load_mode = True
            if key == ord('m'):
                self.load_mode = False

            if key == ord('p'):
                print(self.point_lists)
                
            if key == ord('n'):
                self.get_mask()
                self.polish()
                print('polished!')
                cv2.imshow('final',self.final_img)
                cv2.waitKey(0)
                
            if key == ord('y'):
                self.get_mask_from_png('./mask.png')
                self.polish()
                print('polished!')
                cv2.imshow('final',self.final_img)
                cv2.waitKey(0)
                
            
            
            
            
            
            

        

def get_point_list(O:tuple,S:tuple,B:tuple,I:tuple):
    OX = O[0]
    OY = O[1]
    SX = S[0]
    SY = S[1]
    BX = B[0]
    BY = B[1]
    IX = I[0]
    IY = I[1]
    SIZE_X = OX*2+SX*2+BX
    SIZE_Y = OY*2+SY*2+BY
    OSX = OX+SX
    OSY = OY+SY
    point_lists = [
        np.array([
            [OSX,OY],[SIZE_X-OSX,OY],[OSX,OSY],[SIZE_X-OSX,OSY]
        ],dtype=np.float32),
        np.array([
            [OX,SIZE_Y-OSY],[OX,OSY],[OSX,SIZE_Y-OSY],[OSX,OSY]
        ],dtype=np.float32),
        np.array([
            [SIZE_X-OSX,SIZE_Y-OY],[OSX,SIZE_Y-OY],[SIZE_X-OSX,SIZE_Y-OSY],[OSX,SIZE_Y-OSY]
        ],dtype=np.float32),
        np.array([
            [SIZE_X-OX,OSY],[SIZE_X-OX,SIZE_Y-OSY],[SIZE_X-OSX,OSY],[SIZE_X-OSX,SIZE_Y-OSY]
        ],dtype=np.float32)
    ]
    return point_lists,SIZE_X,SIZE_Y




                


                
        
    
    
    





