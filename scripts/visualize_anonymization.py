import cv2
import os
from tqdm import tqdm
import numpy as np

# Define the file paths
data_dir = 'data/hard'
anonymized_dir = 'output/hard/anonymized_images'
anonymized_bbox_dir = 'output/hard/anonymized_bbox_images'
output_video_path = 'output/hard/video.mp4'

# Define the font and size
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
font_thickness = 2

def create_video(data_dir, anonymized_dir, anonymized_bbox_dir, output_video_path):
    # Check if the anonymized_bbox_dir exists
    bbox_exists = os.path.exists(anonymized_bbox_dir)
    # Get the dimensions of the first image
    first_img = cv2.imread(os.path.join(data_dir, 'cn01', '0000004000_color.jpg'))
    height, width, _ = first_img.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 6.0, (width*2 + 10 if not bbox_exists else width*3 + 20, height*4))

    # Loop over all frames
    for i in tqdm(range(4000, 4000 + 5*75, 5)):
        # Read unanonymized images
        unanonymized_images = []
        for j in range(1, 5):
            img_path = os.path.join(data_dir, f'cn0{j}', f'000000{i}_color.jpg')
            img = cv2.imread(img_path)
            # Add camera number
            img = cv2.putText(img, f'Camera {j}', (10, 50), font, font_scale, font_color, font_thickness)
            unanonymized_images.append(img)
        # Combine unanonymized images
        unanonymized_combined = cv2.vconcat([cv2.vconcat(unanonymized_images[:2]), cv2.vconcat(unanonymized_images[2:])])
        # Add text
        unanonymized_combined = cv2.putText(unanonymized_combined, 'Unanonymized', (10, 20), font, font_scale, font_color, font_thickness)
        
        # Read anonymized images
        anonymized_images = []
        for j in range(1, 5):
            img_path = os.path.join(anonymized_dir, f'Frame_{i}_cn0{j}.jpg')
            img = cv2.imread(img_path)
            # Add camera number
            img = cv2.putText(img, f'Camera {j}', (10, 50), font, font_scale, font_color, font_thickness)
            anonymized_images.append(img)
        # Combine anonymized images
        anonymized_combined = cv2.vconcat([cv2.vconcat(anonymized_images[:2]), cv2.vconcat(anonymized_images[2:])])
        # Add text
        anonymized_combined = cv2.putText(anonymized_combined, 'Anonymized', (10, 20), font, font_scale, font_color, font_thickness)
        
        # Combine unanonymized and anonymized images
        white_line = np.ones((height*4, 10, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([unanonymized_combined, white_line, anonymized_combined])
        
        # If bbox_exists, read anonymized bbox images, combine and add to final combined image
        if bbox_exists:
            # Read anonymized bbox images
            anonymized_bbox_images = []
            for j in range(1, 5):
                img_path = os.path.join(anonymized_bbox_dir, f'Frame_{i}_cn0{j}.jpg')
                img = cv2.imread(img_path)
                # Add camera number
                img = cv2.putText(img, f'Camera {j}', (10, 50), font, font_scale, font_color, font_thickness)
                anonymized_bbox_images.append(img)
            # Combine anonymized bbox images
            anonymized_bbox_combined = cv2.vconcat([cv2.vconcat(anonymized_bbox_images[:2]), cv2.vconcat(anonymized_bbox_images[2:])])
            # Add text
            anonymized_bbox_combined = cv2.putText(anonymized_bbox_combined, 'Anonymized with Bounding Box', (10, 20), font, font_scale, font_color, font_thickness)
            
            # Combine all images
            combined = cv2.hconcat([combined, white_line, anonymized_bbox_combined])
        
        # Write the frame
        out.write(combined)
    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

# Create the video
create_video(data_dir, anonymized_dir, anonymized_bbox_dir, output_video_path)
