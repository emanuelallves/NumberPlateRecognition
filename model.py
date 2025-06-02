from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data = 'C:\Portfolio\ComputerVision\ObjectDetection\PlateRecognation\Project\config.yaml',
                      epochs = 1)