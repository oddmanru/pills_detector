import numpy as np



'''
2022/11/25 This is the 2nd edition of UnpackObjects class
In wake of the implementation for uppacking the detection objects of YOLOv5, the 2nd
edition adds a new function which unpacks the detections of MobileNetSSD 

This code snippet provides two methods to unpack the detected objects structure 
done by YOLOv5 or YOLOv3. The unpack_numpy(), e.g, is believed a faster way to 
return the unpack objects while the unpack_loop() method is the legacy but more 
intuitive method that goes through 25200 loops to find out the results  
'''
class UnpackObjects:
    #
    def __init__(self, classes=None,  confident_score=0.4, score_thresh=0.25, delta_timer=None):
        # classes = CLASSES
        self.classes = classes
        self.confident_score = confident_score
        self.score_thresh = score_thresh
        self.delta_timer = delta_timer

        
        
    def unpack_numpy(self, detected_objects, width_ratio=1.6, height_ratio=1.6):
        '''
        There was a minor change in this code snippet which makes it more 
        readable, i.e applied the satisfied_list to feed into the conf_and_scores array
        '''
    
        # Obtain the matrix containing confidences and the scores for 80 classes
        conf_and_scores = detected_objects[:, 4:]
        # print(f"[conf and scores]: {conf_and_scores}")
        # print("-----------------")
        # Filter out this list which contains the row numbers that satisfy the confidences>=0.4
        satisfied_list = np.argwhere(conf_and_scores[:, 0]>=self.confident_score).flatten()
        # print(f"[satisfied list]: {satisfied_list}")
        # print("-----------------")
        # Return the matrix that contains all the  

        # confident_scores = conf_and_scores[np.argwhere(conf_and_scores[:, 0]>= self.confident_score).flatten(), 1:]
        confident_scores = conf_and_scores[satisfied_list, 1:]

        # print(f"[confident scores]: {confident_scores}")
        # Filter out a new matrix that satifies all the scores are greater than 0.25
        confident_higher_scores = confident_scores * (confident_scores > self.score_thresh)
        # print(f"[confident and higher scores]: {confident_higher_scores}")
        
        # Obtain the final result list
        final_row_list = satisfied_list[np.nonzero(np.max(confident_higher_scores, axis=1))].flatten()
        # print(f"[final rows]: {final_row_list}")
        
        # Obtaint the corresponding final classIDs
        # However, because the final_row_list might have rows that contain 0, a np.sum() test
        # has to be done to fiter it out
        non_zero_rows = np.argwhere(np.sum(confident_higher_scores, axis=1)>0).flatten()
        # print(f"[non zero rows]: {non_zero_rows}")
        classIDs = np.argmax(confident_higher_scores[non_zero_rows], axis=1)
        # print(f"[classIDs]: {classIDs}")
        '''
        array([[0, 2],
               [1, 5],
               [5, 0]], dtype=int64)
        '''
        results_arr = np.vstack([final_row_list, classIDs]).T
        # print(f"[results_arr]: {results_arr}")
        # print(f"[rows] {results_arr[:, 0]}" )
        
        # Obtain the bboxes
        orig_bboxes = detected_objects[results_arr[:,0], :4]
        # print(f"[orig_bboxes]: {orig_bboxes}")
        # [centerX, centerY, w, h]
        new_width = orig_bboxes[:, 2] / 2
        new_height = orig_bboxes[:, 3] / 2 
        
        # Create a new_width and new_height array 
        w_and_h = np.vstack([new_width, new_height]).T
        wh_rows = w_and_h.shape[0]
        wh_cols = w_and_h.shape[1]
        arr_length = wh_rows * wh_cols
        zeros_arr = np.zeros(arr_length).reshape(wh_rows, wh_cols)
        offset_arr = np.hstack([w_and_h, zeros_arr])
        # print(f"[offset array]: {offset_arr}")
        tmp_bboxes = orig_bboxes - offset_arr
        ratio_arr = np.array([width_ratio, height_ratio, width_ratio, height_ratio])
        # Obtain the output bboxes
        output_bboxes = (tmp_bboxes * ratio_arr).astype("int")
        
        
        # Obtain the confidences
        output_confidences = detected_objects[results_arr[:, 0], 4] 
        # Obtain the classIDs
        output_classIds = results_arr[:, 1]
        
        return output_bboxes, output_confidences, output_classIds
        
        
    def unpack_yolov5(self, detections, width_ratio=1.6, height_ratio=1.6):
        '''
        unpack_yolov5 is a updated wrapper of the function unpack_numpy, 
        which was badly named. The purpose of doing this is to makes it 
        more consistent with other functions such as unpack_mobilenetSSD

        '''
        bboxes = None
        confidences = None
        classIDs = None
        bboxes, confidences, classIDs = self.unpack_numpy(self, detections, 
                                                        width_ratio=width_ratio, 
                                                        height_ratio=height_ratio)
        return bboxes, confidences, classIDs


    def unpack_loop(self, detected_objects, CLASSES=None, width_ratio=2, height_ratio= 2, delta_timer = None):
        
        '''
        This code snippet is aimed at creating a generator to unwrap the 
        bboxes and their corresponding confidences and classIDs. 
        the purpose of this experiment is to see if it can reduce the computing 
        time since generator is a memory sufficient method. 
        '''
        # for i in range(len(detected_objects)):
        # for i in np.arange(0, len(detected_objects)):
        boxes= []
        confidences= []
        classIDs = []
        for detected_object in detected_objects:
            

            # YOLOv5
            # print(f"[yolov5 detections] detections.shape: {detections.shape}")
            confidence = detected_object[4]
            # print(f"[yolov5 detection] confidence: {confidence}")
            # print(f"[yolov5 detection] confidence shape: {confidence.shape}")
            if confidence >= 0.4:
                

                # Figure out the maximum score, shape= (80, )
                class_scores = detected_object[5:]


                # maxIdx is a (y, x) tuple, where x is the row's number
                # which represents the class ids
                # cv2 methods follows the (h, w) order
                # minMax_timer.start()

                _, _, _, maxIdx = cv2.minMaxLoc(class_scores)
                # print(f"[max idx]: {maxIdx[1]}")
                # minMax_timer.stop()
                # minMax_timer.update()


                # Return the row's number which represents the class IDs
                classID = maxIdx[1]
                # E.g, class_scores[0]: 0.9168790578842163
                if class_scores[classID] > 0.25:
                    
                    # Obtain the label name
                    label = CLASSES[classID]

                    # if label =="car" or label == "bus" or label == "person" or label=="truck":

                    # scale the bounding box coordinates back relative to the
                    # size of the image. 
                    # ATTN: keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height

                    # ATTN! The original bboxes that YOLOv5 returned are the real
                    # format of the coords, which are different than the YOLOv3 which returns
                    # the normalized coords. 
                    # [original box] [322.7322   220.8373    18.013727  29.702003]
                    orig_box = detected_object[0:4]
                    # print(f"[original box] {orig_box}")


                    # box= detections[0:4] * np.array([w, h, w, h])
                    # Testing... The following is the original code
                    #x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    (centerX, centerY, width, height)= orig_box.astype('int')

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    # ATTN! YOLOv5 returns the real bboxes, so it needs to 
                    # restore the original bbox by multiplying the width and height ratio
                    tlX= int((centerX- (width/2)) * width_ratio)
                    tlY= int((centerY- (height/2)) * height_ratio)

                    # Here is a test
                    # brX= int((centerX + (width/2)) * width_ratio)
                    # brY= int((centerX + (height/2)) * height_ratio)

                    width_new = int(width * width_ratio)
                    height_new = int(height * height_ratio)
                    # print(f"[width_new] {width_new}")
                    # print(f"[height_new] {height_new}")

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([tlX, tlY, width_new, height_new])
                    confidences.append(float(confidence))
                    classIDs.append(classID)


        # Create the generator-based objects
        return boxes, confidences, classIDs
        # yield boxes
        # yield confidences
        # yield classIDs


    def unpack_mobilenetSSD(self, detections, height, width, conf_thresh = 0.5):
        '''
        detections: the detected array returned by the MobileNetSSD object detector
        height, width: the height and width of the frame originally was
        '''
        bboxes = None
        confidences = None
        classIDs = None
        # Return the array which satisfied with the confidence when it is greater than 0.5
        sati_arr = detections[0,0, np.argwhere(detections[0,0, :, 2]>=conf_thresh).flatten()]
        # Obtain the bboxes, confidences and classIDs
        bboxes = sati_arr[:, 3:] * np.array([width, height, width, height])
        confidences = sati_arr[:, 2]
        classIDs = sati_arr[:, 1]

        return bboxes, confidences, classIDs 