from flask import request, jsonify
from werkzeug.utils import secure_filename
import cv2
import os

ALLOWED_EXTENSIONS = {'mp4', 'wmv', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_file_upload(upload_folder):
    # Check if 'file' is part of the request
    if 'file' not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename).replace("\\", "/")
        temp_filepath = os.path.join(upload_folder, "temp_" + filename).replace("\\", "/")
        file.save(temp_filepath)

        # Reduce quality and save the video
        compress_video(temp_filepath, filepath)

        # Remove the temporary file
        os.remove(temp_filepath)
        print(filepath)
        return filepath  # Return filepath for successful upload
    else:
        return jsonify({"message": "Invalid file type"}), 400

def compress_video(input_path, output_path, width=1220, height=780, fps=20, quality=50):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output file
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and resize each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to reduce quality
        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)

    # Release resources
    cap.release()
    out.release()
