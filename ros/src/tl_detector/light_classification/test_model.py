
# coding: utf-8

# In[1]:

#from styx_msgs.msg import TrafficLight 

import numpy as np
import os
import sys
#import cv2
#import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

from collections import defaultdict
from io import StringIO

from utilities import label_map_util
from utilities import visualization_utils as vis_util




class TLClassifier(object):
    
    def __init__(self, *args):

        #print("Classifier launched in site mode : ", args[0])

        #self.current_light = TrafficLight.RED
        self.current_light = 0
        #cwd = os.path.dirname(os.path.realpath(__file__))

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        base_path = '/home/ashis/ashis/Udacity/SDC/Term_3/TL_Model_Test/CarND-Capstone-Ashis-master/ros/src/tl_detector/light_classification/' #os.path.dirname(os.path.abspath(__file__))
        #MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
        MODEL_NAME = 'ssd_mobilenet_tl'
        PATH_TO_CKPT = os.path.join(base_path, MODEL_NAME, 'frozen_inference_graph.pb')
        #CHUNK_SIZE = 10485760  # 10MB
        #PATH_TO_CHUNKS = os.path.join(base_path, MODEL_NAME, 'chunks')

        #print('checkpoint', PATH_TO_CKPT)
        #print('chunks', PATH_TO_CHUNKS)

        # If the frozen model does not exist trying creating it from file chunks
        #if not os.path.exists(PATH_TO_CKPT):  #(MODEL_NAME + '/frozen_inference_graph.pb'):
        #    print("frozen inference graph not found - building from chunks")
        #    output = open(PATH_TO_CKPT, 'wb')
        #    chunks = os.listdir(PATH_TO_CHUNKS)
        #    chunks.sort()
        #    for fname in chunks:
        #        fpath = os.path.join(PATH_TO_CHUNKS, fname)
        #        with open(fpath, 'rb') as fileobj:
        #            for chunk in iter(lambda: fileobj.read(CHUNK_SIZE), b''):
        #                output.write(chunk)
        #    output.close()

        # Load label map
        
        PATH_TO_LABELS = os.path.join(base_path, 'data', 'traffic_lights_label_map.pbtxt')
        NUM_CLASSES = 14
        
        #label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        label_map = label_map_util.load_labelmap('/home/ashis/ashis/Udacity/SDC/Term_3/TL_Model_Test/CarND-Capstone-Ashis-master/ros/src/tl_detector/light_classification/ssd_mobilenet_tl/traffic_lights_label_map.pbtxt')
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        
        self.category_index = label_map_util.create_category_index(categories)

        # Build network
        self.detection_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # https://github.com/tensorflow/tensorflow/issues/6698

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        print("Classifier initialisation complete!")


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #image = cv2.imread(str(image))
        image = Image.open(image)
        image_np_expanded = np.expand_dims(image, axis=0)
        print ('======= image_np_expanded ',image_np_expanded.shape)

        # Perform network inference
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # Check the detections. If it has a good score
        # then set the current light to the detected label. The
        # first one is alwasy the best (they are returned sorted 
        # in score order).
        # Note that we have trained for 14 categories, including
        # left/right arrows etc. Here we are only looking for 
        # standard red, yellow and green light and ignore others.
        #print ('===== category_index ',self.category_index)
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > .05:
                print('scores  ', classes[i])
                #classname = self.category_index[classes[i]]['name']

                if classes[i] == 1:
                    self.current_light = 2 #TrafficLight.GREEN
                elif classes[i] == 7:
                    self.current_light = 1 #TrafficLight.YELLOW
                elif classes[i] == 2:
                    self.current_light = 0 #TrafficLight.RED
                else:
                    self.current_light = 4 #TrafficLight.UNKNOWN
            else:
                self.current_light = 4
            break

        print ('==== current Detected light ', self.current_light)
        return self.current_light


if __name__ == "__main__":

    #TLClassifier()
    img_path = '/home/ashis/ashis/Udacity/SDC/Term_3/TL_Model_Test/CarND-Capstone-master/ros/src/tl_detector/light_classification/data/test_images_sim/test_images/1506839567.02_.png'
    tl_clf =TLClassifier()
    current_light = tl_clf.get_classification(img_path)
    print ('=== current_light ',current_light)







