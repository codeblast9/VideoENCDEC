import cv2
import numpy as np

# --- SETTINGS ---
input_video = "input.mp4"
output_video = "input-enc.mp4"
text = "567"
target_frame_index = 50  # We will only hide the message on this specific frame
# ----------------

cap = cv2.VideoCapture(input_video)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Use mp4v codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"Encoding hidden message '{text}' on frame {target_frame_index} only...")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # ONLY modify the frame if it matches our target index
    if frame_count == target_frame_index:
        print(f"Applying watermark to frame {frame_count}...")
        
        # 1. Create a black mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 2. Calculate position for Bottom-Right Corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3  # Adjusted size
        thickness = 5
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_w = text_size[0]
        text_h = text_size[1]
        
        # Coordinates: Width minus text width (minus padding), Height minus padding
        text_x = width - text_w - 50 
        text_y = height - 50
        
        cv2.putText(mask, text, (text_x, text_y), font, font_scale, (255), thickness)
        
        # 3. Embed into Blue channel
        b, g, r = cv2.split(frame)
        
        # Add the mask value to the blue channel
        # We multiply mask by a small factor to make it subtle but recoverable
        b = cv2.add(b, (mask // 255 * 20).astype(np.uint8)) 
        
        frame = cv2.merge((b, g, r))

    # Write the frame (modified or original) to output
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"Done! Saved as {output_video}. Total frames: {frame_count}")