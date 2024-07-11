# Hot Dog / Not Hot Dog

Inspired by the Silicon Valley (S4E4) [episode](https://www.youtube.com/watch?v=ACmydtFDTGs).

## The hypothesis

Given a training dataset, see if the system can successfully predict if an image contains a hot dog - or not.

## Motivations

This code was developed as an introductory project to determine how to train an AI model to detect
an object in a picture. Beyond the Silicon Valley reference, this sort of problem feels like it's
straightforward enough to be developed and tested with reasonable hardware and software requirements.

If you're not familiar with training an AI model, you may think that the coding is the most important
part. Surprisingly, most of the "difficult" code has already been written. And, in many cases, that
code has been made available as Open Source.

If the difficult coding is already complete, you may be wondering: what's left to do? This README will
attempt to highlight the end-to-end process involved in building an AI project.


## Development Steps

### Step 1: Identify the model to be used

For object detection, I performed a number of web searches, and even a few ChatGPT queries. Based on
that information, I identified the top two candidate models to use for my project:

- Detectron2: https://github.com/facebookresearch/detectron2
- YOLO (You only look once): https://github.com/ultralytics/ultralytics

Based upon additional criteria, I selected Detectron2 for the initial work. However, I made a note to
limit my dependencies on this model, so if things change in the future, I can replace Detectron2 without
major rework in the deployed application.


### Step 2: Identify a Dataset to train the model

It's surprising how hard this step is. Most people assume that one can simply pull a bunch of data off
off the internet, and they're ready to start training. There are a number of reasons why this isn't true,
but perhaps the simplist deals with licensing and how to limit exposure.

For this exercise, I found a dataset that met my particular needs. However, this is really left as an
exercise to the reader.

If you're feeling particularly stuck, you may want to consider the following sites:
- roboflow.com:  https://universe.roboflow.com/workspace-2eqzv/hot-dog-detection/dataset/2
- kaggle.com:  https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog

Be sure that you're comfortable with the terms and conditions associate with the dataset.


### Step 3: Learn how to use the model

So this step is pretty interesting. In the application development world, there's extensive
documentation and tutorials available for just about every topic. This guidance is thorough, available
from many sources, and usually of very high quality.

When it comes to AI models, my experience has been that the documentation is... sparse. Most of the
docs are incomplete and lacking real world examples of how to use them. They're also written from the
perspective of AI researchers, so it may be necesasry to look up the meaning of several new terms.
It might also be necessary to dive into the code to uncover the information you're after.

Fortunately, I found 3rd party materials which helped me better understand how to use the model.
(You can find additional information which I used in the Appendix below.)

By the way, it shouldn't be a surprise that the documentation isn't perfect: AI is an area of rapid
change, and what's the state-of-the-art today may be completely irrelevant tomorrow. Besides, all of this
is being made available to you for the low, low cost of zero. So instead of attempting to compare
AI model documentation with other software engineering topics, it may be best to consider model
access somewhat similar to being a beta customer for a really cool future product. (As always,
your mileage may vary. Enjoy.)


## Step 4: Define the Architecture

In the AI/ML project that I've worked on so far, there are essentially two different environments
to consider:
- The training environment / architecture
- The deployed application runtime / architecture

For the training environment, one of the most difficult requirements is access to a GPU-based machine.
And specifically, nearly everything I've used has required access to an **NVIDIA** GPU-based machine.
(And to be even more specific, the platform must support **CUDA** - https://en.wikipedia.org/wiki/CUDA )

If you don't happen to have one of these machines, a number of alternatives are available, including:

1. Amazon's Sagemaker - free tier: https://aws.amazon.com/sagemaker/
2. Google's Colab: https://colab.research.google.com/

Since the selection of the hardware can be the determining factor in the rest of the architecture,
the detailed elaboration of the architecture is beyond the scope of this document. However, in general,
it should include the following:

1. Access to a GPU-based machine
2. Support for Jupyter Notebook development on the GPU-based machine
3. Object storage (read-write enabled)
4. Optional: ETL parallel processing (if the data is truly voluminous)

As far as a runtime for the finalized application, the following high level architecture will be used:

1. A containerized application.
2. Development language: Python (with Flask for web access)
3. Temporary storage for image upload. (Less than 10GB is fine for development)
4. Appropriate security measures are in place, including things like security groups, firewalls, and the like.


## Step 5: Transform the Data

When selecting the model to be used, it's critical to understand the file format that it expects to find.
In my case, both my was in the YOLO format, but Detectron expected a slightly different format.

In `config.py`, I include a function to transorm YOLO format into a format that Detectron2 accepts. In this
case, the data was small enough to load completely into memory, so a real-time transformation was feasible.
However, if your data set is too big for this, consider an interim ETL (Extract, Transform, and Load) Job
to shape the data before loading.

In general, Detectron2 expects data in a format similar to the following:

```
<root-data-dir>
  |
  -- train
     |
     -- imgs
     -- anns
  -- val
     |
     -- imgs
     -- anns
```
Where:
- `train` - data used for the training phase
- `val` - data used for the validation phase
- `imgs` - the actual images to use in the training process
- `anns` - metadata which includes the bounding boxes in YOLO format

Note: You can read more about the YOLO format here:
https://docs.cogniflow.ai/en/article/how-to-create-a-dataset-for-object-detection-using-the-yolo-labeling-format-1tahk19/

## Step 6: Train the Model (and Validate it too)

## Step 7: Develop the Runtime Code

## Step 8: Test, Tune, and Tune Again

## Step 9: Deploy



Command from tutorial:
`python train.py --device gpu --learning-rate 0.00001 --iterations 5000`



# Appendix

A special thanks to Felipe - the Computer vision engineer - and his great YouTube tutorial.
You can find it here: https://www.youtube.com/watch?v=I7O4ymSDcGw

