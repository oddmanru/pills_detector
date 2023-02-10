# pills_detector
Technically, this is a pills bottle detector which was derived from a custom object detector trained by yolov5. Identifying the pill bottles dedicated for family members is challenging, particularly when people are getting old. E.g, they often forget where the bottle is, or they cannot ensure if the pill bottle is dedicated for someone as the bottles were acquired from the same pharmacy although they were labeled differently. 
Applying one of the state-of-the-art object detectors i.e yolov5 to train a custom object detector might help people address this issue. Therefore, this code snippet is aimed at training and testing a custom object detection process which distinguish different pill bottles even though they sometimes look alike.   

What is different than a regular YOLOv5 detector which typically uses the detect.py code developed by Ultralytics, this code leveraged the OpenCV DNN backend as the DL backbone, therefore an onnx-format weight file was converted from the pytorch's dot-pt-format which was taken by the transfer learning proccess. From this stand point, it is an indepentent app for detecting pill bottles.  

![home_1676003582](https://user-images.githubusercontent.com/99988506/218037589-9625cb12-f613-4b45-b31b-fb9bb86078d5.jpg)
