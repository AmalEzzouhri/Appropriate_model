import numpy as np
import configuration
import cv2

#para
config = configuration.Configuration()

def marge_box(box):
    xmin0,ymin0,xmax0,ymax0=box
    if(ymax0+50>476):
        ymax=476
    else:
        ymax=ymax0+50

    if(ymin0-50<0):
        ymin=0
    else:
        ymin= ymin0-50  

    if(xmax0+50>748):

        xmax=xmax0
    else:
        xmax= xmax0+50

    if(xmin0-50<0):
        xmin=0
    else:
        xmin= xmin0-50 
    return (xmin,ymin,xmax,ymax)

def select_model(bb,config): # return model_id
#def select_model([H,W],config): # return model_id

    xmin,ymin,xmax,ymax = bb

    W = xmax-xmin
    H = ymax-ymin
    dist_list = []

    for ctr in config.clusters_centroid: # ctr_format : (y,x)
        ed=(ctr[0]-H)**2+(ctr[1]-W)**2
        dist_list.append(ed)
   
    (m,i) = min((v,i) for i,v in enumerate(dist_list))
    proposed_h,proposed_w=config.input_shapes[i]
    if W<=proposed_w and H<=proposed_h:     
        return i # id 0 is for the main detector using the entire image
    else:
        return -1  

def normalize_shape(croped_roi,expected_shape): # return new_image,h_padding,w_padding

    h,w,c= croped_roi.shape

    new_image = np.zeros((expected_shape[0], expected_shape[1], c), np.uint8)

    h_padding = (expected_shape[0] - h)//2

    w_padding = (expected_shape[1] - w)//2

    new_image[h_padding:h_padding+h, w_padding:w_padding+w]=croped_roi
       
    return new_image,h_padding,w_padding


cap = cv2.VideoCapture('D:/Amal/renamedFolder/research/object_detection/data/video_000.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = np.ones((5,5),np.uint8)
while(1):
    boxes_detected=[]
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    im2,contours,hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        #cv2.drawContours(frame, contours, -1, 255, 3)

        #find the biggest area
        c = max(contours, key = cv2.contourArea)
        
        

        x,y,w,h = cv2.boundingRect(c)
        # draw the book contour (in green)
        if w>30 and h>30 and w<500:
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            boxes_detected.append((x,y,x+w,y+h))
        
        if(len(contours)>1):
            sorteddata=sorted(contours, key = cv2.contourArea)
            # print(sorteddata[1][1])
            (x1,y1,w1,h1) = cv2.boundingRect(sorteddata[1])
            print(x1,y1,w1,h1)
            
            if (w1>30 and h1>30):#intersects((x,y,x+w,y+h),(x1,y1,x1+w1,y1+h1)) ==True
                print(x1,y1)    
                
                cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,255),3)
                boxes_detected.append((x1,y1,x1+w1,y1+h1))
                
        
    if (len(boxes_detected)>1): 
        cpt=0                       
        print("more thqn box")
        for box in boxes_detected:
            #cropped image
            box_merge_50=marge_box(box)
            print("box_merge_50 : ",box_merge_50)
            image=frame[box_merge_50[1]:box_merge_50[3],box_merge_50[0]:box_merge_50[2]]

            #model selection
            model_id=select_model(box_merge_50,config)

            #image normalization    
            expected_shape=config.input_shapes[model_id]
            new_image,h_padding,w_padding=normalize_shape(image,expected_shape)
            print("model_selection with shape :",image.shape)

            print("model_selected:",model_id)
            cv2.imshow('croped_reagion_normalization'+str(cpt),new_image)
            cv2.imshow('croped_reagion'+str(cpt),image)    
            cpt=cpt+1
        
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',opening)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
