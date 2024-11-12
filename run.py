import cv2
import numpy as np
import pyautogui

# Capture a screenshot of the entire screen
screenshot = pyautogui.screenshot()
screen_image_color = np.array(screenshot)  # Keep the colored version

# Load the target image in color
target_image = cv2.imread('images/target.png')

# Set similarity threshold (e.g., 0.9 to be more strict)
threshold = 0.9

# Loop through different scales to account for size differences
found_locations = []
scales = np.linspace(0.8, 1.2, 10)  # Scale from 80% to 120% in 10 steps
for scale in scales:
    # Resize the target image
    resized_target = cv2.resize(target_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if resized_target.shape[0] > screen_image_color.shape[0] or resized_target.shape[1] > screen_image_color.shape[1]:
        # Skip if the resized target is larger than the screenshot
        continue

    # Template matching using the colored version
    result = cv2.matchTemplate(screen_image_color, resized_target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Store found locations if they exceed the threshold
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):  # locations[::-1] gives coordinates (x, y) instead of (y, x)
        bottom_right = (pt[0] + resized_target.shape[1], pt[1] + resized_target.shape[0])
        # Add only if the result is a reasonable match (in case of false positives due to scaling)
        if max_val >= threshold:
            found_locations.append((pt, bottom_right))

# Draw rectangles around all matched fragments on the colored image
for top_left, bottom_right in found_locations:
    cv2.rectangle(screen_image_color, top_left, bottom_right, (0, 255, 0), 2)

# Display the image with marked matches in color
cv2.imshow('Multiple Matches', screen_image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display all coordinates
print("Found matches (top left corner):", [top_left for top_left, _ in found_locations])