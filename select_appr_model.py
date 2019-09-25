import cv2
from . import configuration
from . import detectors
import math


detections=detect_by_gmm(frame)
config = configuration.Configuration()

def detect_process(detections):
	absolute_detections = []	
	
	for bb in detections:
		
		croped_roi = frame[bb[0]:bb[1], bb[2]:bb[3]]
		model_id=select_model(bb)
		relative_detections,h_padding,w_padding=detect_by_cnn(croped_roi,model_id)
		abs_det = relative_to_absolute(relative_detections,bb,h_padding,w_padding)
		absolute_detections.append(abs_det)
	return absolute_detections

def select_model(bb,config):
	
	xmin,xmax,ymin,ymax = bb
	
	W = xmax-xmin
	H = ymax-ymin
	dist_list = []

	for ctr in config.clusters_centroid:
		ed=(ctr[0]-w)^2+(ctr[1]-H)^2
		dist_list.append(ed)

	(m,i) = min((v,i) for i,v in enumerate(dist_list))
	proposed_h,proposed_w,c=config.input_shapes[i+1]
	if W<=proposed_w and H<=proposed_h:		
		return i+1 # id 0 is for the main detector using the entire image
	else:
		return 0

def normalize_shape(croped_roi,expected_shape):
	h,w,c=croped_roi.shape
	new_image = np.zeros(( expected_shape[0], expected_shape[1], c), np.uint8)
	
	h_padding = int(h - expected_shape[0])/2
	w_padding = int(w - expected_shape[1])/2

	new_image[h_padding:h_padding+h, w_padding:w_padding+w]=croped_roi
	return new_image,h_padding,w_padding
	
def detect_by_cnn(croped_roi,model_id,config):

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

def relative_to_absolute(relative_detections,bb,h_padding,w_padding):
	
	abs_dets=[]
	for rel_det in xrange relative_detections:
		xmin,xmax,ymin,ymax = rel_det	

		abs_xmin = xmin + w_padding + bb[0]
		abs_xmax = xmax + w_padding + bb[1]

		abs_ymin = ymin + h_padding + bb[2]
		abs_ymax = ymax + h_padding + bb[3]

		abs_det = (abs_xmin,abs_xmax,abs_ymin,abs_ymax)
		abs_dets.append()
	return abs_det
