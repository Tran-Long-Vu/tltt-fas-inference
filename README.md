# Project: VHT-facial-anti-spoof


## Congifuration Environment

    python 3.11
    pytorch 2.2.1
    torchvision 0.17.1
    cuda 11.8

Run: "pip install -r requirements.txt"

## Before test

Its best to have the folders /model and /data ready before cloning the repository.

For mock data & models. Download models & data on: https://drive.google.com/drive/folders/1BSyfksbBj5dvB_0Q6e5pJiwyCjx_z68p?usp=drive_link

Git clone the repository.

Put the /data and /model folders in the correct order in the repository. Then cd to the repo.

### Model Directory:

- model
  - scrfd.onnx
  - rn18-fas.onnx
  - mv3-fas.onnx
  - rn18-fas-ckp.pth

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
    - video_benchmark
      - 0
        - real.mp4
### Run file:

python test.py

- Default test is printing attack.
- Test script is for .onnx model.
- Available test types: 'printing', 'replay', 'live'.
- Change INFERENCE_TYPE in configs.py to the correct type to test.
- To test checkpoint change TEST_FROM_CHECKPOINT in configs.py to True.

# Additional work to be added:
- log metrics on pyplot sklearn
