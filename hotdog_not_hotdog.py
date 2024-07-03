
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import cv2

# Hot dog / not hot dog
import argparse

def load_config(mz_name: str, pretrained_model: str):
    config = get_cfg()
    config.merge_from_file(model_zoo.get_config_file(mz_name))
    config.MODEL.WEIGHTS = pretrained_model
    config.MODEL.DEVICE = "cpu"
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="An image that might contain a hot dog")
    parser.add_argument("model", help="A pretrained model to use")
    args = parser.parse_args()

    predict_config = load_config("COCO-Detection/retinanet_R_101_FPN_3x.yaml", args.model)
    predictor = DefaultPredictor(predict_config)

    image = cv2.imread(args.image)
    raw_results = predictor(image)
    results = raw_results["instances"]

    # See if it meets our threshold for a valid image
    threshold = 0.50 #.75

    # Show the results
    prediction_results = results.pred_classes.tolist()
    prediction_scores  = results.scores.tolist()
    prediction_boxes   = results.pred_boxes

    for idx, bbox_raw in enumerate(prediction_boxes):
        bbox = bbox_raw.tolist()
        score = prediction_scores[idx]
        prediction = prediction_results[idx]

        if score > threshold:
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)

    cv2.imshow('image', image)
    cv2.waitKey(0)



