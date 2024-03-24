import json
import os
from zipfile import ZipFile
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

SIZE = 768  # Size for resizing and cropping
PLOT_BBOX = False  # Option to plot bounding boxes

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def resize_and_crop_image(image_path, bboxes, target_size = SIZE):
    # Read the image with OpenCV
    img = cv2.imread(image_path)
    input_height, input_width = img.shape[:2]
    
    # Convert bboxes to imgaug format
    bboxes_ia = BoundingBoxesOnImage([
        BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3]) for bbox in bboxes
    ], shape=img.shape)
    
    # Determine resizing strategy based on the aspect ratio
    if input_width < input_height:
        seq = iaa.Sequential([
            iaa.Resize({"height": "keep-aspect-ratio", "width": target_size}),
            iaa.CropToFixedSize(width=target_size, height=target_size, position="center")
        ])
    else:
        seq = iaa.Sequential([
            iaa.Resize({"height": target_size, "width": "keep-aspect-ratio"}),
            iaa.CropToFixedSize(width=target_size, height=target_size, position="center")
        ])
    
    # Apply the sequence to the image and bounding boxes
    image_aug, bboxes_aug = seq(image=img, bounding_boxes=bboxes_ia)
    
    # Convert imgaug bounding boxes back to the original format
    bboxes_adjusted = [(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1) for bbox in bboxes_aug]
    # modify so they are all within the range.
    bboxes_adjusted = [(max(0, x), max(0, y), w, h) for x, y, w, h in bboxes_adjusted]


    # Convert the OpenCV image format back to PIL Image format for consistency with the original functions
    image_aug_pil = Image.fromarray(cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB))
    
    # Return the PIL image and the first (and only) adjusted bounding box
    return image_aug_pil, bboxes_adjusted




# Extract zip files if not already extracted
def extract_zip(zip_file, extract_path):
    if not os.path.exists(extract_path):
        with ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            print(f"{zip_file} extracted.")
    else:
        print(f"{zip_file} already extracted.")

def plot_bboxes_fn(image, bboxes):
    if PLOT_BBOX:
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            # make it all positive
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), max(0, x_max), max(0, y_max)
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
    return image

# Downloaded files
train_zip_file = "train2017.zip"
val_zip_file = "val2017.zip"
test_zip_file = "test2017.zip"
trainval_annotation_zip_file = "annotations_trainval2017.zip"
test_annotation_zip_file = "image_info_test2017.zip"

# Extract zip files
extract_zip(train_zip_file, 'train2017')
extract_zip(val_zip_file, 'val2017')
extract_zip(test_zip_file, 'test2017')
extract_zip(trainval_annotation_zip_file, 'annotations')
extract_zip(test_annotation_zip_file, 'annotations')

# Initialize COCO instances for train, val, and test datasets
train_coco = COCO('annotations/instances_train2017.json')
val_coco = COCO('annotations/instances_val2017.json')
test_coco = COCO('annotations/image_info_test2017.json')

from tqdm import tqdm
# Preprocess dataset
def preprocess_coco(coco_instance, img_dir, preprocess_path, plot_bbox=False, test_N=3):
    dataset = []
    for i, img_id in tqdm(enumerate(coco_instance.imgs)):
        # if i >= test_N:
        #     break
        img_info = coco_instance.loadImgs(img_id)[0]
        # if img is too small, skip
        if img_info['width'] < 450 or img_info['height'] < 450:
            print("SKIP!!")
            continue
        annotations_ids = coco_instance.getAnnIds(imgIds=img_info['id'])
        annotations = coco_instance.loadAnns(annotations_ids)
        bboxes = []
        category_names = []
        existing_cat = set()
        for annotation in annotations:
            category_name = coco_instance.loadCats(annotation['category_id'])[0]['name']
            bbox = annotation['bbox']
            # if its width or height is out of range, then skip
            if bbox[2] < 75 or bbox[3] < 75:
                continue
            # if the cate
            if category_name in existing_cat:
                continue

            if len(existing_cat) > 4:
                continue
            existing_cat.add(category_name)
            bboxes.append(bbox)
            category_names.append(category_name)
            #print(f"Processing {img_info['file_name']} with {category_name} and bbox {bbox}")
        img_path = os.path.join(img_dir, img_info['file_name'])
        image, bboxes = resize_and_crop_image(img_path, bboxes)
        
        image_with_bbox = plot_bboxes_fn(image.copy(), bboxes)
        if plot_bbox:
            image_with_bbox.save(os.path.join(preprocess_path, f"bbox_{img_info['file_name']}"))
        image.save(os.path.join(preprocess_path, img_info['file_name']))
        prompt = ''
        for category_name, bbox in zip(category_names, bboxes):
            bbox_as_str = ''.join(f"<|{round(coord)}|>" for coord in bbox)
            

            prompt += f"{bbox_as_str} {category_name} "
            
        dataset.append({"img_path": os.path.abspath(os.path.join(preprocess_path, img_info['file_name'])), "prompt": prompt})
    return dataset

# Preprocess train, val, and test datasets
preprocessed_path = "preprocessed_coco"
os.makedirs(preprocessed_path, exist_ok=True)

train_dataset = preprocess_coco(train_coco, 'train2017', preprocessed_path, PLOT_BBOX)
val_dataset = preprocess_coco(val_coco, 'val2017', preprocessed_path, PLOT_BBOX)
# test_dataset = preprocess_coco(test_coco, 'test2017', preprocessed_path, PLOT_BBOX)

# Write datasets to JSON files
with open('train.json', 'w') as train_json_file:
    json.dump(train_dataset, train_json_file)
    print("Train dataset saved to train.json.")
with open('val.json', 'w') as val_json_file:
    json.dump(val_dataset, val_json_file)
    print("Validation dataset saved to val.json.")
# with open('test.json', 'w') as test_json_file:
#     json.dump(test_dataset, test_json_file)
#     print("Test dataset saved to test.json.")
