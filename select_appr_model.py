import cv2
import configuration
# import detectors
import numpy as np

#xml to csv 
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def detect_process(detections,config): # return absolute_detections
	absolute_detections = []	
	
	for bb in detections:
		
		croped_roi = frame[bb[0]:bb[1], bb[2]:bb[3]]
		model_id=select_model(bb,config)
		relative_detections,h_padding,w_padding=detect_by_cnn(croped_roi,model_id,config)
		abs_det = relative_to_absolute(relative_detections,bb,h_padding,w_padding)
		absolute_detections.append(abs_det)
	return absolute_detections

def select_model(bb,config): # return model_id
	
	xmin,ymin,xmax,ymax = bb
	
	W = xmax-xmin
	H = ymax-ymin
	dist_list = []

	for ctr in config.clusters_centroid: # ctr_format : (y,x)
		ed=(ctr[0]-H)^2+(ctr[1]-W)^2
		dist_list.append(ed)

	(m,i) = min((v,i) for i,v in enumerate(dist_list))
	proposed_h,proposed_w=config.input_shapes[i]
	if W<=proposed_w and H<=proposed_h:		
		return i # id 0 is for the main detector using the entire image
	else:
		return len(config.input_shapes)

def normalize_shape(croped_roi,expected_shape): # return new_image,h_padding,w_padding

	h,w,c= croped_roi.shape

	new_image = np.zeros((expected_shape[0], expected_shape[1], c), np.uint8)

	h_padding = int(expected_shape[0] - h)/2

	w_padding = int(expected_shape[1] - w)/2

	new_image[h_padding:h_padding+h, w_padding:w_padding+w]=croped_roi

	
	cv2.imshow("original", croped_roi)
	cv2.imshow("normalized", new_image)
	# cv2.imwrite("normalized.png", new_image)
	key = cv2.waitKey(3000) & 0xFF

	
	return new_image,h_padding,w_padding



def relative_to_absolute(relative_detections,bb,h_padding,w_padding): # return absolute_dets
	
	abs_dets=[]

	for rel_det in relative_detections:
		xmin,ymin,xmax,ymax = rel_det	

		abs_xmin = xmin + w_padding + bb[0]
		abs_xmax = xmax + w_padding + bb[2]

		abs_ymin = ymin + h_padding + bb[1]
		abs_ymax = ymax + h_padding + bb[3]

		abs_det = (abs_xmin,abs_ymin,abs_xmax,abs_ymax)
		abs_dets.append(abs_det)
	return abs_dets

def absolute_to_relative(abs_cordinate,bb): # return relative_dets

	input_shapes = [(200,280),(370,380),(480,450),(480,748)]
	clusters_centroid = [(140,164),(224,234),(414,362),(425,531)]
	xmin,ymin,xmax,ymax = bb
	
	W = xmax-xmin
	H = ymax-ymin
	dist_list = []

	for ctr in clusters_centroid:
		ed=(ctr[0]-H)**2+(ctr[1]-W)**2
		dist_list.append(ed)

	(m,i) = min((v,i) for i,v in enumerate(dist_list))
	proposed_h,proposed_w=input_shapes[i]
	
	if W<=proposed_w and H<=proposed_h:		
		model_id =  i 
	else:
		model_id = len(input_shapes)
	

	H_padding = int(input_shapes[model_id][0] - H)/2
	W_padding = int(input_shapes[model_id][1] - W)/2
	
	abs_xmin,abs_ymin,abs_xmax,abs_ymax = abs_cordinate
	
	rel_xmin = abs_xmin + W_padding - xmin
	rel_xmax = abs_xmax + W_padding - xmax
	rel_ymin = abs_ymin + H_padding - ymin
	rel_ymax = abs_ymax + H_padding - ymax

	return rel_xmin,rel_ymin,rel_xmax,rel_ymax

def xml_to_csv(path): # return xml_df
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def detect_by_cnn(croped_roi,model_id,config): # return dets,h_padding,w_padding

	new_image,h_padding,w_padding = normalize_shape(croped_roi,config.input_shapes[model_id])

	if model_id==0:
		dets=detect_by_0(new_image)
	if model_id==1:
		dets=detect_by_1(new_image)	
	if model_id==2:
		dets=detect_by_2(new_image)
	if model_id==3:
		dets=detect_by_3(new_image)

	return dets,h_padding,w_padding

def main():

	# GLOBAL stuff
	# detections=detect_by_gmm(frame)
	config = configuration.Configuration()	
	test_img = cv2.imread('C:/Users/Public/Pictures/Sample Pictures/test.jpg')
	

	# test NORMALIZE INPUT SHAPE
	print("normalize_shape")
	normalize_shape(test_img,config.input_shapes[3])

	# test RELATIVE_to_absolute
	print('relative_to_absolute')
	relative_detections = [[70,170,40,105],[90,160,50,70]]
	bb = [50, 300 , 20 , 340]
	h_padding = 25
	w_padding = 40
	print(relative_to_absolute(relative_detections,bb,h_padding,w_padding))

	# test SELECT model
	print('select_model')
	print(select_model(bb,config))

	# test ABSOLUTE_to_relative
	print('absolute_to_relative')
	abs_cordinate = [70,350,40,415]
	print(absolute_to_relative(abs_cordinate,bb))
	
	# # test XML_to_CSV
	# for directory in ['train','test']:
	# 	image_path = os.path.join(os.getcwd(), format(directory))
	# 	xml_df = xml_to_csv(image_path)
	# 	xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)

	
	# test DETECT_by_cnn


main()

