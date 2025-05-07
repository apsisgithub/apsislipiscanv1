import cv2
import numpy as np
from typing import List, Dict, Any,Tuple

def draw_word_polys(image: np.ndarray, word_polys: List[List]) -> np.ndarray:
    """
    Draws the word polygons from OCR results on the input image.
    
    Parameters:
        image (np.ndarray): The input image.
        words (List[List]): The word polygon list with [x1,y1,x2,y2,x3,y3,x4,y4]

    Returns:
        np.ndarray: The image with drawn word polygons.
    """
    # Make a copy of the image to draw on
    output_image = image.copy()
    
    # Iterate over the words
    for poly in  word_polys:
        poly = np.array(np.array(poly).reshape(-1,2), dtype=np.int32)  # Convert polygon to NumPy array
        # Draw the polygon
        cv2.polylines(output_image, [poly], isClosed=True, color=(0, 255, 0), thickness=1)

    return output_image