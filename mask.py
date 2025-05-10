import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

def create_mask_from_coco(coco_path, masks_dir):
    """
    Convert COCO-style annotations to multi-class masks.
    Each character has its own category_id (1-36).
    """
    os.makedirs(masks_dir, exist_ok=True)
    coco = COCO(coco_path)

    for img_id in tqdm(coco.getImgIds(), desc=f"Processing {os.path.basename(os.path.dirname(coco_path))}"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            cat_id = ann['category_id']
            if cat_id == 0:
                continue  # skip general "char" label
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], color=cat_id)

        # Save mask with same name but .png extension
        mask_path = os.path.join(masks_dir, os.path.splitext(file_name)[0] + '.png')
        Image.fromarray(mask).save(mask_path)

def process_all_splits(base_dir):
    splits = ['train', 'test', 'valid']
    for split in splits:
        coco_json = os.path.join(base_dir, 'images', split, '_annotations.coco.json')
        masks_dir = os.path.join(base_dir, 'masks', split)
        create_mask_from_coco(coco_json, masks_dir)

# Run it
process_all_splits("dataset")
