import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

initial = []
final = []

print("Instructions:")
print("1. Press 'r' once to save the INITIAL frame.")
print("2. Move an object, then press 'r' again to save the FINAL frame.")
print("3. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # ALWAYS show the live camera feed so you know what you are capturing
    cv2.imshow('Live Feed', frame)

    # Listen for keys
    key = cv2.waitKey(1) & 0xFF

    # When 'r' is pressed, capture frames sequentially
    if key == ord('r'):
        if len(initial) == 0:
            initial = frame.copy()  # Use .copy() to prevent reference issues
            print("Initial frame captured!")
        elif len(final) == 0:
            final = frame.copy()
            print('Final move captured! Calculating difference...')
        else:
            initial = frame.copy()
            final = []
            print("Reset: New initial frame captured!")
    elif key == ord('c'):
        initial = []
        final = []
        print("Cleared captured frames.")

    # ONLY calculate and show the difference if BOTH frames exist
    if len(initial) > 0 and len(final) > 0:
        gray1 = cv2.cvtColor(initial, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
        
        # Get the absolute difference
        diff = cv2.absdiff(gray1, gray2)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Remove noise
        diff = cv2.dilate(diff, None, iterations=4)
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        diff = cv2.erode(diff, kernel, iterations=6)


        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours_and_areas = sorted(zip(contours, [cv2.contourArea(c) for c in contours]), key=lambda x: x[1], reverse=True)
        
        # Display the difference map
        cv2.imshow('Difference', diff)

    # Press 'q' to break out of the loop and close the program
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
