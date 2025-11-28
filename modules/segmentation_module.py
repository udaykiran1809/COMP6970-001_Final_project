import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
import os

def ade_palette():
    # ... (paste the full, long ade_palette list here as before) ...
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255], [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 122], [255, 184, 0], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 153], [255, 82, 0], [255, 214, 0], [255, 0, 194], [255, 0, 163], [255, 143, 0], [51, 153, 255], [153, 255, 0], [0, 255, 235], [255, 245, 0], [163, 255, 0], [122, 255, 0], [214, 255, 0], [0, 82, 255], [0, 255, 204], [204, 0, 255], [255, 204, 0], [0, 255, 224], [0, 153, 255], [255, 0, 71], [10, 255, 0], [153, 0, 255]]

# NEW: Standalone utility function for creating color masks
def create_color_mask(pred_seg: np.ndarray) -> Image.Image:
    """Converts a numeric segmentation map to a color-coded PIL Image."""
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    palette_np = np.array(ade_palette())
    for label, color in enumerate(palette_np):
        color_seg[pred_seg == label, :] = color
    return Image.fromarray(color_seg)

class SemanticSegmenter:
    """A class to encapsulate the SegFormer model and processing logic."""
    def __init__(self, model_name="nvidia/segformer-b5-finetuned-ade-640-640", cache_dir=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        print(f"Loading SegFormer model: {self.model_name} to {self.device}...")
        self.processor = SegformerImageProcessor.from_pretrained(self.model_name, cache_dir=cache_dir)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name, cache_dir=cache_dir).to(self.device)

    def get_segmentation_map(self, image: Image.Image) -> np.ndarray:
        # (This method remains the same)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode='bilinear', align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        return pred_seg

    def get_detected_objects(self, pred_seg: np.ndarray) -> dict:
        # (This method remains the same)
        unique_ids = np.unique(pred_seg)
        detected = {}
        for class_id in unique_ids:
            label_name = self.model.config.id2label.get(int(class_id), "unknown")
            detected[int(class_id)] = label_name
        return detected