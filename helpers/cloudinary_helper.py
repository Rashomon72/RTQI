import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from dotenv import load_dotenv
import os
import time


load_dotenv()


def handle_cloudinary_upload(filepath):
    try:
        timestamp = int(time.time())  # Current timestamp in seconds
        filename = os.path.splitext(filepath.split("/")[-1])[0]
        chunk_size_gb = 1
        chunk_size_bytes = chunk_size_gb * 1073741824
        public_id = f"{timestamp}_{filename}"
        cloudinary.config(
            cloud_name="dxky3ao7h",
            api_key="657835628191426",
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True
        )

        upload_result = cloudinary.uploader.upload_large(filepath,
                                                         resource_type="video",
                                                         public_id=public_id,
                                                         chunk_size=chunk_size_bytes,
                                                         eager=[{"audio_codec": "none"}])

        # # Delete the local copy after uploading to cloudinary
        # if os.path.exists(filepath):
        #     os.remove(filepath)
        #     print(f"File '{filepath}' has been deleted successfully.")
        # else:
        #     print(f"File '{filepath}' does not exist.")

        return upload_result["secure_url"]

    except Exception as e:
        print(f"Failed to upload to Cloudinary ¯\_(ツ)_/¯: {e}")
