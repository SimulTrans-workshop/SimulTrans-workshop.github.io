---
layout: main-anchor
title: tutorial
order: 7
collection: pages_2020
---

### Training a TensorFlow-based Transformer Model from Scratch in Docker

We will dive head-first into training a transformer model from scratch using a TensorFlow GPU Docker image.


#### Step 1) Launch TensorFlow GPU Docker Container

Using Docker allows us to spin up a fully contained environment for our training needs. We always recommend using Docker, as it allows ultimate flexibility (and forgiveness) in our training environment. To begin we will open a terminal window and enter the following command to launch our NVIDIA CUDA powered container.
```bash
nvidia-docker run -it -p 6007:6006 -v /data:/datasets tensorflow/tensorflow:nightly-gpu bash
```
Note: A quick description about the key parameters of the above command (if you’re unfamiliar with Docker).

Docker Syntax|	Description
-|-
nvidia-docker run|	The 'docker run' command specifies the container from. Note in this case we use ‘nvidia-docker run’ to utilize CUDA powered NVIDIA GPUs.
-p 6007:6006|	Expose port 6007 [HOST:CONTAINER] for Tensorboard, type localhost:6007 in your browser to view Tensorboard
-v data:/datasets|	Volume tag, This shares the folder '/data' on the host machine to the /datasets folder in the container. Also explained as: -v /[host folder]:/[container folder]
tensorflow/tensorflow:nightly-gpu|	This the docker image that will run. The Docker Hub format is [PUBLISHER]/[IMAGE REPO]:[IMAGE TAG]

#### Step 2) Install git
This may be necessary if you are running a fresh docker container.

`apt-get install git`


#### Step 3) Download Models (Transformer Codebase is included here)
In case you do not have the latest up-to-date codebase for the models, the transformer network model is included here and the devs tend to update it quite frequently.

`mkdir /transformer; cd /transformer`  
`git clone https://github.com/tensorflow/models.git`

#### Step 4) Install Requirements for TensorFlow Models
As a necessary step,this will install the python package requirements for training TensorFlow models.

`# cd to the models dir`  
`pip install --user -r official/requirements.txt`


#### Step 5) Export Pythonpath
Export PYTHONPATH to the folder where the models folder are located on your machine. The command below references where the models are located on our system. Be sure to replace the `/transformer/models` syntax with the data path to the folder where you stored/downloaded your models.

`export PYTHONPATH="$PYTHONPATH:/transformer/models"`

#### Step 6) Download and Preprocess the Dataset
The `data_download.py` command will download and preprocess the training and evaluation WMT datasets. Upon download and extraction, the training data is used to generate for what we will use as `VOCAB_FILE` variables. Effectively, the eval and training strings are tokenized, and the results are processed and saved as TFRecords.

NOTE: ([per the official requirements](https://github.com/tensorflow/models/tree/master/official/transformer)): 1.75GB of compressed data will be downloaded. In total, the raw files (compressed, extracted, and combined files) take up 8.4GB of disk space. The resulting TFRecord and vocabulary files are 722MB. The script takes around 40 minutes to run, with the bulk of the time spent downloading and ~15 minutes spent on preprocessing.

`cd /transformer/models/official/transformer`  
`python data_download.py --data_dir=/datasets/transformer`


#### Step 7) Set Training Variables for the Transformer Model
‘PARAM_SET’

This specifies what model to train. ‘big’ or ‘base’

IMPORTANT NOTE: The ‘big’ model will not work on most consumer grade GPU’s such as RTX 2080 Ti, GTX 1080 Ti. If you need to train the ‘big’ model we recommend a system with at least 48 available GB GPU memory such as a Data Science Workstation equipped with the Quadro RTX 8000’s, or 2 x Qudaro RTX 6000 with NVLink. Alternatively a TITAN RTX Workstation with 2x TITAN RTX (With NVLink Bridge) should also suffice. For this example, we’re using an RTX 2080 Ti, so we select ‘base‘.

PARAM_SET=base
‘DATA_DIR’

This variable should be set to where the training data is located.

DATA_DIR=$root/datasets/datasets/transformer
‘MODEL_DIR’

This variable specifies the model location based on what model is specified in the ‘PARAM_SET’ variable

MODEL_DIR=$root/datasets/datasets/transformer/model_$PARAM_SET
‘VOCAB_FILE’

This variable expresses where the location of the preprocessed vocab files are located.

VOCAB_FILE=$DATA_DIR/vocab.ende.32768
‘EXPORT_DIR’ Export trained transformer model

This will specify the location when/where you export the model in Tensorflow SavedModel format. This is done when using the flag export_dir when training in step 8.

EXPORT_DIR=$root/datasets/datasets/transformer/saved_model



