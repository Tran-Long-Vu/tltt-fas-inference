# Project: VHT-facial-anti-spoof
## Training Facial Anti Spoof model on CVPR23 dataset.

Congifuration Environment

    python 3.11
    pytorch 2.2.1
    torchvision 0.17.1
    cuda 11.8

Run: "pip install -r requirements.txt"

## Pre-training

Git clone the repository.

Download the CVPR23 Dataset.
Save samples in directory: $root/data/datasets/CVPR23/train/
The dataset is split into 80% training and 20% validation.

For mock training & models. Download models & training data on: https://drive.google.com/drive/folders/1BSyfksbBj5dvB_0Q6e5pJiwyCjx_z68p?usp=drive_link

## Training Data Directory:

- data
    - datasets
        - CVPR23
            - Train
                - living
                    - subject
                        - imagename.jpg
                        - imagename.txt
                - spoof
                    - PAs type # printing attack type
                        - subject
                            - imagename.jpg
                            - imagename.txt

### Model Directory:
- model
  - scrfd.onnx
  - rn18-fas.onnx
  - mv3-fas.onnx
  - rn18-fas-ckp.pth
  
### To check if model & data configurations are ready:
- python3 checker.py


## Training

Move to the folder $root of the repository, in separate terminal, run:

- mlflow server --host 127.0.0.1 --port 8080

Then run: 

- python3 train.py

The file /configs/config.py contains all the hyper-parameters used during training.

Saved checkpoints are in directory: /checkpoints/fas-best.ptl

Training metrics are recorded at: http://127.0.0.1:8080

## Testing with hand crawled data:

### Data directory: 
- data
    - datasets
        - Hand-crawl
            - videos
                - 0
                    - videoname.mp4
                - 1
                    - videoname.mp4
            - images
                - 0
                    - imagename.jpg
                - 1                
                    - imagename.jpg

### Run file:

python test.py

# Additional work to be added:
- Training script from .pth checkpoint.
- Converting .pth checkpoint to onnx format.
- Test script based on training checkpoint.
