import cv2 
def resize_for_ocr(image):
  height, width = image.shape[:2]
  if width < 1500:
    scale = 1500 / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = cv2.resize(image, (new_width, new_height))
  return image