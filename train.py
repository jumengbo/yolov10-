from ultralytics import YOLOv10


model = YOLOv10('./ultralytics/cfg/models/v10/yolov10s.yaml')

model.train(data='substation.yaml', epochs=200, batch=32, imgsz=640)

#yolo detect train data=screen.yaml model=yolov10s.yaml epochs=10 batch=8 imgsz=640 device=0