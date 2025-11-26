import cv2
import numpy as np

# Settings
recorded_video = "recorded1.mp4"

cap = cv2.VideoCapture(recorded_video)

# Skip to the middle of the video to get a steady frame
frame_count = 0
target_frame = 30 
extracted_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count == target_frame:
        extracted_frame = frame
        break
    frame_count += 1
cap.release()

if extracted_frame is None:
    print("Error: Could not read recorded video.")
    exit()

print("Processing frame to reveal hidden layers...")

# 1. Split the channels
b, g, r = cv2.split(extracted_frame)

# 2. subtract Green from Blue

diff = cv2.subtract(b, g)

# 3. Enhance Contrast (Thresholding)
contrast_enhanced = cv2.multiply(diff, 10)

# 4. Save and Show
cv2.imwrite("decoded_message.png", contrast_enhanced)

print("---------------------------------------------")
print("SUCCESS: Check 'decoded_message.png'")
print("You should see the number 123 clearly.")
print("---------------------------------------------")

# Optional: Show on screen
cv2.imshow("Hidden Layer Revealed", contrast_enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()