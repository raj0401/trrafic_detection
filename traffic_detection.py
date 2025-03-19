import cv2
import torch
import threading
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO  
from deep_sort_realtime.deepsort_tracker import DeepSort  
from PIL import Image, ImageTk

# Load YOLOv8 Model
model = YOLO('yolov8s.pt')
tracker = DeepSort(max_age=30)
video_source = None
cap = None
stop_tracking = False

def select_file():
    global video_source
    video_source = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if video_source:
        lbl_status.config(text=f"Selected File: {video_source.split('/')[-1]}")

def select_webcam():
    global video_source
    video_source = 0  # Webcam source
    lbl_status.config(text="Selected Source: Webcam")

def start_tracking():
    global stop_tracking
    if video_source is None:
        lbl_status.config(text="Please select a tracking option first!")
        return
    stop_tracking = False
    threading.Thread(target=run_tracking, daemon=True).start()

def stop_tracking_fn():
    global stop_tracking
    stop_tracking = True

def run_tracking():
    global cap, stop_tracking
    cap = cv2.VideoCapture(video_source)
    vehicle_classes = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0}
    tracked_vehicles = {cls: set() for cls in vehicle_classes}
    
    while cap.isOpened():  
        ret, frame = cap.read()  
        if not ret or stop_tracking:
            break  
        
        results = model(frame)
        detections = []  
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                confidence = box.conf[0].item()  
                class_id = int(box.cls[0].item())  
                label = model.names[class_id]  
                
                if label in vehicle_classes:
                    detections.append(([x1, y1, x2, y2], confidence, class_id))
        
        tracks = tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = track.det_class  
            label = model.names[class_id]
            
            if label in tracked_vehicles and track_id not in tracked_vehicles[label]:
                tracked_vehicles[label].add(track_id)
                vehicle_classes[label] += 1  
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        update_counts(vehicle_classes)
        show_frame(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_tracking = True
            break  
    
    cap.release()
    cv2.destroyAllWindows()

def update_counts(vehicle_classes):
    total_count = sum(vehicle_classes.values())
    count_text = "\n".join([f"{vehicle}: {count}" for vehicle, count in vehicle_classes.items()])
    lbl_counts.config(text=f"{count_text}\nTotal Vehicles: {total_count}")

def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((500, 300))
    img_tk = ImageTk.PhotoImage(img)
    lbl_video.img_tk = img_tk
    lbl_video.config(image=img_tk)

# GUI Setup
root = tk.Tk()
root.title("Traffic Analyzer GUI")
root.geometry("600x500")

lbl_status = tk.Label(root, text="Select a tracking option to start", fg="blue")
lbl_status.pack()

frame_options = tk.Frame(root)
frame_options.pack(pady=10)

btn_file = tk.Button(frame_options, text="Choose File", command=select_file)
btn_file.pack(side=tk.LEFT, padx=5)

btn_webcam = tk.Button(frame_options, text="Webcam", command=select_webcam)
btn_webcam.pack(side=tk.LEFT, padx=5)

btn_start = tk.Button(root, text="Start Tracking", command=start_tracking)
btn_start.pack(pady=10)

btn_stop = tk.Button(root, text="Stop Tracking", command=stop_tracking_fn, fg="red")
btn_stop.pack(pady=10)

lbl_video = tk.Label(root)
lbl_video.pack()

lbl_counts = tk.Label(root, text="Vehicle Counts:", font=("Arial", 12))
lbl_counts.pack()

root.mainloop()

