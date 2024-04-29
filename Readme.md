# If you have a csv file for key points annotated for various video , then create a folder named raw_data and place both video and csv file in it
* Run command pip install -r requirements.txt
* Run the yolo gen.ipynb file
* A new folder would be created by names output_frame
* Now train object detection model in roboflow by uploading this output_frame folder there
* Once the traing in completed, in the deploy option select python API
* Replace the Python API with your API & project name also
* Now click install the required packages and run Trajectory.py