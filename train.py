from ultralytics import YOLOv10


model = YOLOv10('./yolov10n.pt')

model.train(data='number.yaml', epochs=500, batch=32, imgsz=640)
