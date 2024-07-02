from ultralytics import YOLO
model = YOLO("/Users/rafael/Desktop/PestTracker2/PestTracker2/best.pt")

model.predict('/Users/rafael/Desktop/ceratitis_video.mp4', save=True, show=True, conf=0.7)