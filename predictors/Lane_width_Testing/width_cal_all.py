import matplotlib
import matplotlib.pyplot as plt
import cv2
import json
import os


matplotlib.use('Agg')

# Function to get points from lanes
def get_lane_points(lane, h_samples):
    points = [(x, y) for x, y in zip(lane, h_samples) if x != -2]
    return points

# Function to annotate and save image
def annotate_and_save(image_path, json_data, output_dir, serial_number):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Coordinates for lanes
    lanes = json_data['lanes']
    h_samples = json_data['h_samples']

    # Get lane points for the two lanes closest to the vehicle
    left_lane_points = get_lane_points(lanes[1], h_samples)
    right_lane_points = get_lane_points(lanes[2], h_samples)

    threshold = 5

    # Get the closest points to the vehicle (or required distance)
    if len(left_lane_points) >= threshold and len(right_lane_points) >= threshold:
        # Get the closest points to the vehicle (or required distance)
        left_closest_point = left_lane_points[-threshold]
        right_closest_point = right_lane_points[-threshold]

        # Calculate the width
        width = abs(left_closest_point[0] - right_closest_point[0])

        # Plot the image and the points
        plt.figure(figsize=(10, 5))
        plt.imshow(image_rgb)
        plt.plot([p[0] for p in left_lane_points], [p[1] for p in left_lane_points], 'go-')
        plt.plot([p[0] for p in right_lane_points], [p[1] for p in right_lane_points], 'go-')

        # Plot the width line
        plt.plot([left_closest_point[0], right_closest_point[0]], [left_closest_point[1], right_closest_point[1]], 'r-', linewidth=2)
        plt.text((left_closest_point[0] + right_closest_point[0]) / 2, left_closest_point[1] - 10, f'Width: {width}px', color='red', fontsize=12, ha='center')

        plt.title('Lane Width Annotation')
        plt.axis('off')  # Turn off axis for cleaner visualization

        # Save annotated image with a serial number
        filename = f"annotated_image_{serial_number}.jpg"
        # filename = "test.jpg"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory
    else:
        print(f"Not enough points in lanes for image {image_path}")

# Path to the annotation file
ann_path = r"C:\Users\Faculty\Downloads\test-images\test_annotations.0.txt"

# Output directory for saving annotated images
output_dir = r"C:\Users\Faculty\Downloads\test-images\width-result"
os.makedirs(output_dir, exist_ok=True)

# Read the annotation file
with open(ann_path, 'r') as f:
    # Initialize serial number counter
    serial_number = 1

    # Read each line (assuming each line is a JSON object)
    for line in f:
        # Parse JSON from each line
        json_data = json.loads(line)

        # Root path to prepend to image paths
        root_path = r"C:\Users\Faculty\Downloads\test-images"

        # Get image path
        image_path = os.path.join(root_path, json_data['raw_file'])
        # image_path = "C:/Users/swath/Desktop/indian-village-road-tree-201186376-transformed.jpg"

        print(image_path)

        # Annotate and save the image with serial number
        annotate_and_save(image_path, json_data, output_dir, serial_number)

        # Increment serial number for the next image
        serial_number += 1

print("Annotation and saving complete.")
