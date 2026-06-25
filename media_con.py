import cv2
import time
import pyautogui
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Cooldown settings (in seconds)
COOLDOWN_TIME = 2.0
last_trigger_time = 0

print("AI Media Controller Started. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Run YOLOv8 inference
    results = model(frame, stream=True)
    current_time = time.time()
    
    phone_detected = False
    cup_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Get properties: Confidence score and Bounding Box coordinates
            confidence = float(box.conf[0]) * 100  # Convert to percentage
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate object dimensions (Width and Height in pixels)
            obj_width = x2 - x1
            obj_height = y2 - y1

            # Target only cell phones, cups, and bottles
            if class_name in ["cell phone", "cup", "bottle"]:
                
                # Determine colors and actions based on class
                if class_name == "cell phone":
                    phone_detected = True
                    color = (255, 0, 0) # Blue
                    action_text = "Action: MUTE"
                else:
                    cup_detected = True
                    color = (0, 255, 0) # Green
                    action_text = "Action: PLAY/PAUSE"

                # 1. Draw the bounding box around the object
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 2. Prepare the properties text strings
                label_name = f"Obj: {class_name.upper()} ({confidence:.1f}%)"
                label_dims = f"Size: {obj_width}x{obj_height} px"
                label_coords = f"Pos: X:{x1} Y:{y1}"
                
                # 3. Overlay the properties text stack next to/above the object
                # We stack them vertically by adding 18-20 pixels to the Y coordinate for each line
                cv2.putText(frame, label_name, (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, label_dims, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, label_coords, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, action_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Action Trigger Logic (with Cooldown)
    if current_time - last_trigger_time > COOLDOWN_TIME:
        if phone_detected:
            print(f"Triggered: Phone detected. Width: {obj_width}px, Height: {obj_height}px")
            pyautogui.press('volumemute')
            last_trigger_time = current_time
            
        elif cup_detected:
            print(f"Triggered: Cup/Bottle detected. Width: {obj_width}px, Height: {obj_height}px")
            pyautogui.press('playpause')
            last_trigger_time = current_time

    # System Status Bar
    time_left = COOLDOWN_TIME - (current_time - last_trigger_time)
    if time_left > 0:
        status_text = f"Automation Cooldown: {time_left:.1f}s"
        status_color = (0, 0, 255) 
    else:
        status_text = "System Ready for Gestures"
        status_color = (0, 255, 0) 
        
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Show the webcam frame
    cv2.imshow("AI Property & Dimension Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()