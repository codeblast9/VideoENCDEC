import cv2
import numpy as np

# Settings
input_video = "a.mp4"
output_video = "a-enc.mp4"
text = "123"

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Use standard mp4v codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("Encoding hidden message... (This puts '123' in the Blue channel)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Create a black mask with the text "123"
    mask = np.zeros((height, width), dtype=np.uint8)
    # Position text in center, large font size
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 10, 20)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(mask, text, (text_x, text_y), font, 10, (255), 20)
    
    # 2. Embed the watermark into the BLUE channel only
   
    
    b, g, r = cv2.split(frame)
    
    # Use numpy to add safely (avoiding overflow)
    b = cv2.add(b, (mask // 255 * 15).astype(np.uint8)) 
    
    # Merge back
    watermarked_frame = cv2.merge((b, g, r))
    out.write(watermarked_frame)

cap.release()
out.release()
print(f"Done! Saved as {output_video}. The '123' is hidden inside the blue color.")