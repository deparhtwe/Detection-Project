
# Step-1 , need to download custom pre-trained model from roboflow 

# train1 to train-3 is for weapons, train-4 to train-5 are for fight 


# from roboflow import Roboflow
#Weapon Model

# rf = Roboflow(api_key="9Z1SEkwXAzXfEhziRp4E")
# project = rf.workspace("weapon-b3vyl").project("weapon_detection_for_final")
# version = project.version(3)
# dataset = version.download("yolov8")

#Fight Model



# from roboflow import Roboflow
# rf = Roboflow(api_key="9Z1SEkwXAzXfEhziRp4E")
# project = rf.workspace("weapon-b3vyl").project("fight-detection-7xdy7-mvwch")
# version = project.version(1)
# dataset = version.download("yolov8")
                



# Step-2 , download yolov8n.pt as in data.yml to train the model,  use this command
#  yolo task=detect model=train model=yolov8n.pt data=C:\Project\Weapons_and_Violence_Detection\Spy-camera--main\weapon_detection_for_final-3\data.yaml epochs=20 imgsz=640
# and then run this command
# yolo predict model=/home/vix/Desktop/Testing/runs/detect/train-5/weights/best.pt conf=0.5 source=fight/test/images



#To run quickly
#for fight, weapons detection

# python ip_webcam_viewer.py 192.168.1.3:8080 --detect-all --imgsz 320 --process-every 1 --stream-width 640 --stream-height 480 

#for cubic detection
#python cubic.py 192.168.150.44:8080 --imgsz 320 --detect-width 416 --process-every 1


#  ip_webcam_viewer.py 192.168.150.47:8080 --detect-all --database-url postgresql://user:password@localhost:5432/your_db



#to change the confidence level in line 956;(default=0.4)

#to update the output result in line 493(detected_type="both")