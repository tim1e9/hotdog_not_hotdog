import argparse
from train_model import train_model_with_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="COCO-Detection/retinanet_R_101_FPN_3x.yaml",
                         help="The model config file, as specified in the Detectron2 model zoo")
    parser.add_argument("--dataset_dir", default="./input", help="The base directory for the dataset to be used")
    parser.add_argument("--output_dir", default="./output", help="The output directory for the trained model")
    parser.add_argument("--use_gpu", default=False, help="Specify whether to use a GPU or a CPU")
    parser.add_argument("--learning_rate", default=0.0001, help="The learning rate to be used while training")
    parser.add_argument("--batch_size", default=4, help="The batch to be used within an iteration")
    parser.add_argument("--num_iterations", default=5000, help="The base directory for the dataset to be used")
    parser.add_argument("--checkpoint_interval", default=500, help="After this many iterations, save a checkpoint")



    args = parser.parse_args()
    print(f"About to train a model with the following arguments: {args}")
    train_model_with_params(args.model_name, args.dataset_dir, args.output_dir, args.use_gpu,
                            args.learning_rate, args.batch_size, args.num_iterations,
                            args.checkpoint_interval)
    print("Done")
    
