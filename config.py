import os
import cv2
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog


# Load default configuration information, but allow for overrides
def load_config_for_model(model_name: str, output_dir: str, use_gpu: bool, num_classes: int,
                      learning_rate: float, batch_size: int, num_iterations: int,
                      checkpoint_interval: int):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))

    # Based on the model, load the original, pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = 'gpu'

    if use_gpu == True:
        cfg.MODEL.DEVICE = 'gpu'
    else:
        cfg.MODEL.DEVICE = 'cpu'

    # Set up the tuples for training and validation
    cfg.DATASETS.TRAIN = ("train")
    cfg.DATASETS.VAL = ("val")
    cfg.DATASETS.TEST = ()

    # Found on the web. Set to an empty array to ensure the learning rate doesn't decay. Um, okay.
    cfg.SOLVER.STEPS = []

    # Based on values passed in, set additional config settings
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = num_iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_interval

    # Some defaults which may be tweaked in the future
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.DATALOADER.NUM_WORKERS = 2 # Default is 4
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    return cfg


# Convert the metadata from Yolo format to Detectron bounding box format
def get_bounding_boxes( lines: str, image_width: int, image_height: int ):
    bounding_boxes = []
    for _, cur_line in enumerate(lines):
        # Extract the components of the string
        label, center_x, center_y, label_width, label_height = cur_line.split(" ")

        x = int((float(center_x) - (float(label_width) / 2)) * image_width)
        y = int((float(center_y) - (float(label_height) / 2)) * image_height)
        w = int(float(label_width) * image_width)
        h = int(float(label_height) * image_height)

        cur_box = {
            "bbox": [x, y, w, h],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": int(label)
        }
        bounding_boxes.append(cur_box)
    
    return bounding_boxes

def get_image_metadata(image_dir: str, metadata_dir: str):
    # Iterate across all files in the md dir, and attempt to locate
    # one in the image dir too.
    image_metadata = []
    for index, md_filename in enumerate(os.listdir(metadata_dir)):
        cur_md = {}
        image_filename = os.path.join(image_dir, md_filename[:-4] + '.jpg')
        height, width = cv2.imread(image_filename).shape[:2]

        # Set some of the image basics
        cur_md['file_name'] = image_filename
        cur_md['image_id'] = index
        cur_md['height'] = height
        cur_md['width'] = width

        # Load all of the bounding boxes from the file
        with open(os.path.join(metadata_dir, md_filename)) as f:
            lines = f.read().splitlines()

        bboxes = get_bounding_boxes(lines, width, height)
        cur_md['annotations'] = bboxes

        image_metadata.append(cur_md)

    return image_metadata

# Register a single phase of the datasets (e.g. training, validation, etc)
# def register_dataset(dataset_dir: str, phase: str, all_classes):
#     image_dir = os.path.join(dataset_dir, phase, 'imgs')
#     metadata_dir = os.path.join(dataset_dir, phase, 'anns')
#     DatasetCatalog.register(phase, lambda: get_image_metadata(image_dir, metadata_dir))
#     MetadataCatalog.get(phase).set(thing_classes=all_classes)


# Register the training and validation datasets
def register_all_datasets(dataset_dir: str, classes_filename: str):
    full_classes_filename = os.path.join(dataset_dir, classes_filename)
    with open(full_classes_filename, 'r') as f:
        all_classes = f.read().splitlines()    

    for d in ['train', 'val']:
        DatasetCatalog.register(d, lambda d=d: get_image_metadata(os.path.join(dataset_dir, d, 'imgs'),
                                                                  os.path.join(dataset_dir, d, 'anns')))
        MetadataCatalog.get(d).set(thing_classes=all_classes)


    # # Register training data
    # register_dataset(dataset_dir, 'train', all_classes)

    # # Register validation data
    # register_dataset(dataset_dir, 'val', all_classes)

    # Return the total number of classes
    return len(all_classes)

