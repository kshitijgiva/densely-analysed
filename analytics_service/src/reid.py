import os
from pathlib import Path
import numpy as np
import torch
import torchreid
import gdown
from torchreid import reid_model_factory

from config import REID_MODEL_NAME, REID_MODEL_CHECKPOINT, REID_WEIGHTS_DIR, _select_device


class OSNetReID:
    """Self-hosted person re-identification embeddings using OSNet (torchreid).

    Produces a 512-dim appearance embedding per person crop, entirely on-device -
    no crops or embeddings leave the local machine.
    """

    def __init__(self, model_name=REID_MODEL_NAME, checkpoint_name=REID_MODEL_CHECKPOINT,
                 weights_dir=REID_WEIGHTS_DIR):
        self.device = _select_device()
        self.embedding_dimension = 512
        weights_path = self._ensure_weights(checkpoint_name, weights_dir)
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=weights_path,
            device=self.device,
            verbose=False,
        )

    @staticmethod
    def _ensure_weights(checkpoint_name, weights_dir):
        """Download the market1501-trained OSNet checkpoint on first run."""
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, checkpoint_name)
        if not os.path.isfile(weights_path):
            url = reid_model_factory.get_model_url(Path(checkpoint_name))
            if url is None:
                raise ValueError(f"No pretrained URL known for checkpoint '{checkpoint_name}'")
            print(f"Downloading OSNet re-id weights ({checkpoint_name})...")
            gdown.download(url, weights_path, quiet=False)
        return weights_path

    def extract_features(self, image):
        """Extract an L2-normalized re-id embedding from a person crop (BGR numpy array)."""
        try:
            if image is None or image.size == 0 or image.shape[0] < 20 or image.shape[1] < 10:
                return None

            rgb_img = image[:, :, ::-1]  # BGR -> RGB
            with torch.no_grad():
                features = self.extractor([rgb_img])[0]

            features = features.cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(features)
            if norm == 0:
                return None
            return features / norm
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return None
