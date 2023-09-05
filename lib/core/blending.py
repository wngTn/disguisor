import cv2
import numpy as np
import copy
from core.poisson import blend
import torch

SKIN_COLOR_W = [255, 255, 255]
SKIN_COLOR_BEIGE = [134, 138, 149]
BUFFER = 12

def get_bboxes(image):
    """
    Returns the bounding boxes of each image
    [x_min, y_min, x_max, y_max]
    """
    
    # Threshold image to get binary mask of non-white pixels
    mask = image[:, :, 0] != 1

    # Get coordinates of non-white pixels
    coords = np.column_stack(np.where(mask))

    # Check if there are any non-white pixels
    if coords.size == 0:
        return np.array([None, None, None, None])

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return np.array([x_min, y_min, x_max, y_max])



def get_mask_and_adjusted_face(cropped_face):
    """
    Returns the mask and adjusted face with skin color
    """
    mask = np.zeros((cropped_face.shape[0], cropped_face.shape[1]), np.uint8)
    cropped_face_bg = cropped_face.copy()
    # Sets the background color of the source to a skin color
    black_region = (cropped_face_bg[:, :, :3] <= [10, 10, 10]).all(axis=2)
    cropped_face_bg[black_region, :3] = SKIN_COLOR_BEIGE
    mask[~black_region] = 255
    # Ensure buffer at the edge
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0
    return mask, cropped_face_bg

def process_face(face_image, num_pixels):
    if np.sum((face_image[:, :, :3] != 1).any(2)) < num_pixels:
        return None
    
    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2BGRA)
    return face_image

def get_cropped_regions(face_image, y_min, y_max, x_min, x_max):
    cropped_face = face_image[
        max(y_min - BUFFER, 0): min(y_max + BUFFER, face_image.shape[0]),
        max(x_min - BUFFER, 0): min(x_max + BUFFER, face_image.shape[1])
    ]
    return cropped_face

def blend_faces_into_background(face_lists, background_images, num_pixels, alpha_value):
    """
    Generates the image where all the faces are blended into the background image.
    """
    
    if len(face_lists) == 0:
        return background_images, np.array([])

    face_lists = torch.stack(face_lists).cpu().detach().numpy()
    bboxes = [[] for _ in background_images]

    for img_i, background_image in enumerate(background_images):
        for f_i, face_image in enumerate(face_lists[:, img_i]):
            
            processed_face = process_face(face_image, num_pixels)
            if processed_face is None:
                continue
            
            x_min, y_min, x_max, y_max = get_bboxes(processed_face)

            # Update bounding boxes
            bboxes[img_i].append(np.array([x_min, y_min, x_max - x_min, y_max - y_min]))

            if x_min is not None:
                cropped_face = get_cropped_regions(processed_face, y_min, y_max, x_min, x_max)
                cropped_target = get_cropped_regions(background_image, y_min, y_max, x_min, x_max)
                
                mask, cropped_face_bg = get_mask_and_adjusted_face(cropped_face)
                result = blend(cropped_face_bg[:, :, :3], mask, cropped_target, alpha_value)

                # Update the background image
                background_image[
                    max(y_min - BUFFER, 0): min(y_max + BUFFER, face_image.shape[0]),
                    max(x_min - BUFFER, 0): min(x_max + BUFFER, face_image.shape[1])
                ] = result

    return background_images, np.array(bboxes, dtype=object)