import os
import config
from detectron2.engine import DefaultTrainer
from notme_loss import ValidationLoss 

# Use reasonable defaults
def train_model_with_params(model_name: str, dataset_dir: str, output_dir: str, use_gpu: bool = True,
                            learning_rate: float = 0.0001, batch_size: int = 4,
                            num_iterations: int = 5000, checkpoint_interval: int = 500):
    class_count = config.register_all_datasets(dataset_dir, "classnames.txt")

    # It seems odd to first register datasets and then load config.
    # Consider changing this
    train_configuration = config.load_config_for_model(model_name=model_name, output_dir=output_dir, use_gpu=use_gpu, 
                                                       num_classes=class_count, learning_rate=learning_rate,
                                                       batch_size=batch_size, num_iterations=num_iterations,
                                                       checkpoint_interval=checkpoint_interval)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    trainer = DefaultTrainer(train_configuration)

    #......................
    # Special thanks to Felipe for this (https://www.youtube.com/@ComputerVisionEngineer)
    # Register the custom validation loss object as a hook to the trainer
    validation_loss = ValidationLoss(train_configuration)
    trainer.register_hooks([validation_loss])
    # Swap the positions of the evaluation and checkpointing hooks so that the validation loss is logged correctly
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    #......................

    trainer.resume_or_load(resume=False)
    trainer.train()
    print("Done")