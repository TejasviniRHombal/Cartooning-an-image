#Method 1

import cv2
import os

# Full path to your image
image_path = "/Users/tejasvinirhombal/Desktop/Cartooning-an-image/image.jpg"

# Check file existence
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at: {image_path}")

img = cv2.imread(image_path)

# Verify image read correctly
if img is None:
    raise SystemExit("❌ OpenCV could not open the image. Check file path and extension.")

# Continue your processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(gray, 255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 9, 9)
color = cv2.bilateralFilter(img, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)

cv2.imwrite("cartoon_output.jpg", cartoon)
print("✅ Cartoon image saved as cartoon_output.jpg")

# Open the cartoon in Preview (macOS)
os.system("open cartoon_output.jpg")


#Method 2

import cv2
import numpy as np

def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

# Read the image
img = cv2.imread('your_image.jpg')

# Color quantization
quantized_img = color_quantization(img, k=9)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median blur, detect edges
blurred = cv2.medianBlur(gray, 7)
edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

# Combine quantized image with edge mask
cartoon_img = cv2.bitwise_and(quantized_img, quantized_img, mask=edges)

# Show or save the result
cv2.imshow('Cartoonized Image', cartoon_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Method 3

import cv2

# Read and scale down the image
img = cv2.imread('your_image.jpg')
img = cv2.pyrDown(cv2.imread('your_image.jpg'))

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply median blur
blurred = cv2.medianBlur(gray, 5)

# Use a Laplacian filter for edge detection
edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=5)

# Invert the color of edges
inverted_edges = 255-edges 

# Combine original image with edges
cartoon_img = cv2.bitwise_or(img, img, mask=inverted_edges)
 
# Show or save the result
cv2.imshow('Cartoonized Image', cartoon_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Method 4

import cv2

# Reading, smoothing, and edge detection all in one line
cv2.imshow('Cartoonized Image', cv2.bitwise_and(cv2.bilateralFilter(cv2.imread('your_image.jpg'), d=9, sigmaColor=300, sigmaSpace=300), cv2.bilateralFilter(cv2.imread('your_image.jpg'), d=9, sigmaColor=300, sigmaSpace=300), mask=cv2.Canny(cv2.imread('your_image.jpg'), 100, 150)))
cv2.waitKey(0)
cv2.destroyAllWindows()
