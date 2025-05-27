import cv2
import os


def get_video_file_name(directory):
    # Define common video file extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a video extension
        if filename.lower().endswith(video_extensions):
            return filename  # Return the first video file found

    return None  # Return None if no video file is found


def extract_frames_per_minute(uploads_folder, folder_name):
    # Create output directory for frames
    video_name = get_video_file_name(uploads_folder)
    video_path = os.path.join(uploads_folder, video_name).replace("\\", "/")
    frames_folder = os.path.join(
        uploads_folder, folder_name).replace("\\", "/")

    os.makedirs(frames_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    frames_per_minute = int(fps * 5)  # Calculate frames per minute
    success, frame = video.read()
    count = 0
    temp = 0  # for debugging

    while success:
        # Check if it's time to save the frame (one per minute)
        if count % frames_per_minute == 0:
            # Minute timestamp for file name
            frame_time = int(count / fps) // 5
            frame_filename = f"frame_{frame_time}min.jpg"
            frame_path = os.path.join(frames_folder, frame_filename)
            cv2.imwrite(frame_path, frame)  # Save frame
            temp = temp + 1 # for debugging
        

        # Read next frame
        success, frame = video.read()
        count += 1

    print("Number of frames extracted: ", temp)
    # Release video and return the folder path
    video.release()
    return frames_folder
