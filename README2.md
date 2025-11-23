# Bibliographical Research (Inference / OCR Script)

This section explains the theoretical background and the implementation details behind the OCR (Optical Character Recognition) script that takes an input image, extracts individual handwritten characters, and classifies them using the previously trained ResNet model.

## Purpose of the Script

The goal of this script is to perform **inference** — that is, using a previously trained neural network to recognize handwritten digits and letters in a real-world image.  
It loads:

- an **input image** containing handwritten characters,
- a **trained ResNet model** (`model.keras`),

and outputs:

- the predicted characters,
- annotated bounding boxes,
- confidence scores.

## OCR Pipeline Overview

OCR (Optical Character Recognition) generally involves three stages:

1. Load image
2. Convert to grayscale
3. Blur and detect edges
4. Find contours
5. Extract character regions
6. Binarize, resize, and normalize characters
7. Pass them through the trained ResNet
8. Draw predictions on the image
9. Display results

## Processing pipeline

The script first loads the input image and prepares it:

```python
# Grayscale conversion
# Removes color information, which handwriting does not need.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
# Reduces noise and smooths the edges for better contour detection.
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# edge detection using Canny's algorithm
edged = cv2.Canny(blurred, 30, 150)

# Contour Detection (Segmentation)
# Contours represent detected shapes in the binary image:
# Only outermost contours are extracted (cv2.RETR_EXTERNAL).
cnts = cv2.findContours(...)
cnts = sort_contours(cnts, method="left-to-right")[0]

# Character Extraction
# Each contour’s bounding box is analyzed:
(x, y, w, h) = cv2.boundingRect(c)

# ROI extraction
# The Region Of Interest (the character) is isolated:
roi = gray[y:y+h, x:x+w]

# Binary threshold
# A mix of Otsu’s thresholding and inversion is applied:
thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Normalization to Model Format
# The trained ResNet expects 32×32 grayscale images, so all ROIs are resized and padded:
padded = cv2.resize(padded, (32, 32))
padded = padded.astype("float32") / 255.0
padded = np.expand_dims(padded, axis=-1)

# Predicting Characters
preds = model.predict(chars)

# Interpreting Predictions
# For each predicted character argmax() selects the class with highest probability
```
