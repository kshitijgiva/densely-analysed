"""
Uploads the pipeline's local heatmap PNG to Cloudinary and returns its public
URL, so /overview can serve a hosted image instead of a local file path.
Credentials come from the environment (see .env.example) - never pass them on
the command line or commit them.
"""
import os

import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

_configured = False


def _configure():
    global _configured
    if _configured:
        return
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME")
    api_key = os.environ.get("CLOUDINARY_API_KEY")
    api_secret = os.environ.get("CLOUDINARY_API_SECRET")
    if not (cloud_name and api_key and api_secret):
        raise RuntimeError(
            "CLOUDINARY_CLOUD_NAME/CLOUDINARY_API_KEY/CLOUDINARY_API_SECRET must be "
            "set (see .env.example) to upload a heatmap."
        )
    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True)
    _configured = True


def upload_heatmap(image_path, store_id, camera_id):
    """Uploads image_path, overwriting the same public_id every call so a
    store/camera's heatmap URL stays stable across pipeline re-runs instead
    of accumulating a new asset each time. Returns the https URL."""
    _configure()
    result = cloudinary.uploader.upload(
        image_path,
        public_id=f"heatmaps/{store_id}_{camera_id}",
        overwrite=True,
        invalidate=True,
        resource_type="image",
    )
    return result["secure_url"]
