import cv2
import numpy as np
import pytesseract




# Point this to where you just installed Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' 
# ---------------------

# ... rest of your code ...
# --- SETTINGS ---
recorded_video = "recorded2.mp4" 

# If you are on Windows, you might need to specify the path to tesseract.exe manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ----------------

cap = cv2.VideoCapture(recorded_video)

if not cap.isOpened():
    print(f"Error: Could not open {recorded_video}")
    exit()

print("Scanning video for hidden text...")

best_score = 0
best_frame_img = None
best_frame_idx = -1

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Split channels
    b, g, r = cv2.split(frame)

    # 2. Subtract Green from Blue (Isolates the blue signal)
    diff = cv2.subtract(b, g)

    # 3. Calculate Score (Brightness sum)
    score = np.sum(diff)

    # 4. Keep the frame with the strongest signal (Brightest hidden text)
    if score > best_score:
        best_score = score
        best_frame_idx = frame_count
        
        # Save the raw difference for processing
        best_frame_img = diff

    frame_count += 1

cap.release()

# --- OCR PROCESSING ---
if best_frame_img is not None:
    print(f"Signal detected at frame {best_frame_idx}. Processing text...")
    
    # 1. Enhance Contrast (Make text white, background black)
    # We multiply by a large factor to brighten the faint hidden text
    enhanced = cv2.multiply(best_frame_img, 15)
    
    # 2. Noise Removal (Thresholding)
    # This removes the grainy noise from the phone camera, leaving only sharp text
    _, binary_img = cv2.threshold(enhanced, 80, 255, cv2.THRESH_BINARY)
    
    # Optional: Dilate slightly to make thin text thicker for the OCR engine
    kernel = np.ones((2,2), np.uint8)
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)

    # 3. Read Text using Pytesseract
    # --psm 6 assumes a single block of text
    try:
        detected_text = pytesseract.image_to_string(binary_img, config='--psm 6')
        
        # Clean up result (remove empty lines)
        detected_text = detected_text.strip()
        
        print("\n" + "="*40)
        print(f"HIDDEN MESSAGE FOUND: {detected_text}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"OCR Error: {e}")
        print("Tip: Make sure Tesseract is installed and path is configured.")

    # 4. Show the image meant for the machine
    output_filename = "decoded_text_image.png"
    cv2.imwrite(output_filename, binary_img)
    
    # Resize for display if too big
    h, w = binary_img.shape
    if w > 1024:
        scale = 1024 / w
        binary_img = cv2.resize(binary_img, (int(w*scale), int(h*scale)))

    cv2.imshow("What the computer sees", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Error: No hidden message detected in the video.")