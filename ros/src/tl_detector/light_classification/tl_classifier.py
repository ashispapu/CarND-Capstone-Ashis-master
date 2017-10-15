from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO

from utilities import label_map_util
from utilities import visualization_utils as vis_util

#from light_classification.utilities import label_map_util
#from light_classification.utilities import visualization_utils as vis_util

#import cv2

#Testing
#from matplotlib import pyplot as plt


class TLClassifier(object):
    
    def __init__(self, *args):


        #self.current_light = TrafficLight.RED
        self.current_light = 0
        #cwd = os.path.dirname(os.path.realpath(__file__))

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        #base_path = os.path.dirname(os.path.abspath(__file__))
        MODEL_NAME = 'ssd_mobilenet_tl/'
        PATH_TO_CKPT = os.path.join('light_classification/', MODEL_NAME, 'frozen_inference_graph.pb')
        

        # Load label map
        
        PATH_TO_LABELS = os.path.join('light_classification/', MODEL_NAME, 'traffic_lights_label_map.pbtxt')
        NUM_CLASSES = 14
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
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

        print("Classifier initialisation completed !")


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        image_np_expanded = np.expand_dims(image, axis=0)

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
