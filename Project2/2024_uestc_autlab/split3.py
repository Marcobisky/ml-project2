# import json
# import os
# import random


# def split_annotations(
#     input_annotations, train_annotations_path, test_annotations_path, val_annotations_path,
#     train_ratio=0.6, test_ratio=0.2, seed=42
# ):
#     """
#     Splits a COCO-style annotations JSON file into training, testing, and validation sets.

#     Args:
#         input_annotations (str): Path to the input annotations JSON file.
#         train_annotations_path (str): Path to save training annotations JSON file.
#         test_annotations_path (str): Path to save testing annotations JSON file.
#         val_annotations_path (str): Path to save validation annotations JSON file.
#         train_ratio (float): Ratio of the dataset to use for training.
#         test_ratio (float): Ratio of the dataset to use for testing.
#         seed (int): Random seed for reproducibility.
#     """
#     # Set random seed
#     random.seed(seed)

#     # Load the input annotations
#     with open(input_annotations, "r") as f:
#         annotations = json.load(f)

#     # Split images into train, test, and validation sets
#     images = annotations["images"]
#     random.shuffle(images)
#     train_size = int(len(images) * train_ratio)
#     test_size = int(len(images) * test_ratio)

#     train_images = images[:train_size]
#     test_images = images[train_size:train_size + test_size]
#     val_images = images[train_size + test_size:]

#     # Get image IDs for train, test, and validation sets
#     train_image_ids = {img["id"] for img in train_images}
#     test_image_ids = {img["id"] for img in test_images}
#     val_image_ids = {img["id"] for img in val_images}

#     # Split annotations based on image IDs
#     train_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] in train_image_ids]
#     test_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] in test_image_ids]
#     val_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] in val_image_ids]

#     # Create training, testing, and validation annotations dictionaries
#     train_data = {
#         "info": annotations.get("info", {}),
#         "licenses": annotations.get("licenses", []),
#         "images": train_images,
#         "annotations": train_annotations,
#         "categories": annotations.get("categories", [])
#     }

#     test_data = {
#         "info": annotations.get("info", {}),
#         "licenses": annotations.get("licenses", []),
#         "images": test_images,
#         "annotations": test_annotations,
#         "categories": annotations.get("categories", [])
#     }

#     val_data = {
#         "info": annotations.get("info", {}),
#         "licenses": annotations.get("licenses", []),
#         "images": val_images,
#         "annotations": val_annotations,
#         "categories": annotations.get("categories", [])
#     }

#     # Save the split annotation files
#     with open(train_annotations_path, "w") as f:
#         json.dump(train_data, f, indent=4)

#     with open(test_annotations_path, "w") as f:
#         json.dump(test_data, f, indent=4)

#     with open(val_annotations_path, "w") as f:
#         json.dump(val_data, f, indent=4)

#     print(f"Training, testing, and validation sets created:")
#     print(f" - Training set: {len(train_images)} images, {len(train_annotations)} annotations")
#     print(f" - Testing set: {len(test_images)} images, {len(test_annotations)} annotations")
#     print(f" - Validation set: {len(val_images)} images, {len(val_annotations)} annotations")


# if __name__ == "__main__":
#     # Input paths
#     input_annotations = "./Lab4/coco/annotations.json"  # Input annotations file

#     # Output paths for training, testing, and validation annotations
#     train_annotations_path = "./Project2/2024_uestc_autlab/data/data_coco_train/annotations.json"
#     test_annotations_path = "./Project2/2024_uestc_autlab/data/data_coco_test/annotations.json"
#     val_annotations_path = "./Project2/2024_uestc_autlab/data/data_coco_valid/annotations.json"

#     # Split the annotations
#     split_annotations(
#         input_annotations,
#         train_annotations_path,
#         test_annotations_path,
#         val_annotations_path
#     )


import json
import os
import random
from collections import defaultdict

def split_annotations(
    input_annotations, 
    train_annotations_path, 
    test_annotations_path, 
    val_annotations_path,
    train_ratio=0.6, 
    test_ratio=0.2, 
    seed=42
):
    """
    Splits a COCO-style annotations JSON file into training, testing, and validation sets
    while maintaining class balance.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the input annotations
    with open(input_annotations, "r") as f:
        annotations = json.load(f)
    
    # Group annotations by category
    category_annotations = defaultdict(list)
    image_to_anns = defaultdict(list)
    
    # First pass: group annotations by image and category
    for ann in annotations["annotations"]:
        category_annotations[ann["category_id"]].append(ann)
        image_to_anns[ann["image_id"]].append(ann)
    
    # Create sets to store image IDs for each split
    train_image_ids = set()
    test_image_ids = set()
    val_image_ids = set()
    
    # Split annotations for each category maintaining ratios
    for category_id, category_anns in category_annotations.items():
        random.shuffle(category_anns)
        
        n_total = len(category_anns)
        n_train = int(n_total * train_ratio)
        n_test = int(n_total * test_ratio)
        
        # Add image IDs to respective splits
        for ann in category_anns[:n_train]:
            train_image_ids.add(ann["image_id"])
        for ann in category_anns[n_train:n_train + n_test]:
            test_image_ids.add(ann["image_id"])
        for ann in category_anns[n_train + n_test:]:
            val_image_ids.add(ann["image_id"])
    
    # Create the split datasets
    image_dict = {img["id"]: img for img in annotations["images"]}
    
    splits = {
        "train": (train_image_ids, train_annotations_path),
        "test": (test_image_ids, test_annotations_path),
        "val": (val_image_ids, val_annotations_path)
    }
    
    # Create and save each split
    for split_name, (image_ids, output_path) in splits.items():
        split_images = [image_dict[img_id] for img_id in image_ids]
        split_annotations = [
            ann for ann in annotations["annotations"]
            if ann["image_id"] in image_ids
        ]
        
        split_data = {
            "info": annotations.get("info", {}),
            "licenses": annotations.get("licenses", []),
            "images": split_images,
            "annotations": split_annotations,
            "categories": annotations["categories"]
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the split
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=4)
        
        print(f"{split_name} set:")
        print(f" - {len(split_images)} images")
        print(f" - {len(split_annotations)} annotations")
        
        # Print class distribution
        class_dist = defaultdict(int)
        for ann in split_annotations:
            class_dist[ann["category_id"]] += 1
        print(f" - Class distribution:", dict(class_dist))
        print()

def verify_splits(train_path, test_path, val_path):
    """Verify that the splits are correct and print statistics."""
    splits = {
        "train": train_path,
        "test": test_path,
        "val": val_path
    }
    
    for name, path in splits.items():
        with open(path) as f:
            data = json.load(f)
            
        # Verify image-annotation consistency
        image_ids = set(img["id"] for img in data["images"])
        ann_image_ids = set(ann["image_id"] for ann in data["annotations"])
        
        print(f"\n{name} split verification:")
        print(f"Images: {len(image_ids)}")
        print(f"Unique images in annotations: {len(ann_image_ids)}")
        print(f"All annotation images exist: {ann_image_ids.issubset(image_ids)}")
        
        # Print class distribution
        class_dist = defaultdict(int)
        for ann in data["annotations"]:
            class_dist[ann["category_id"]] += 1
        print(f"Class distribution: {dict(class_dist)}")

if __name__ == "__main__":
    # Input and output paths
    input_annotations = "./Lab4/coco/annotations.json"
    train_annotations_path = "./Project2/2024_uestc_autlab/data/data_coco_train/annotations.json"
    test_annotations_path = "./Project2/2024_uestc_autlab/data/data_coco_test/annotations.json"
    val_annotations_path = "./Project2/2024_uestc_autlab/data/data_coco_valid/annotations.json"
    
    # Split the annotations
    split_annotations(
        input_annotations,
        train_annotations_path,
        test_annotations_path,
        val_annotations_path
    )
    
    # Verify the splits
    verify_splits(train_annotations_path, test_annotations_path, val_annotations_path)