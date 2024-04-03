from configs.config import *
from libs import *

from data_script.video_dataset import VideoDataset
from data_script.image_dataset import ImageDataset

from engines.face_detector import FaceDetector
from engines.liveness_detection import LivenessDetection

from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort
import torchvision.transforms as tfs

import sklearn.metrics as metrics
import pandas as pd

if __name__ == '__main__':
    training_dataset = ImageDataset(TRAIN_DATASET,
                                    PATH_TO_PRINTING_DATASET,
                                    MODEL_BACKBONE,
                                    augment = 'train',
                                    split= 'train'
                                    )
    
    val_dataset = ImageDataset(     TRAIN_DATASET,
                                    PATH_TO_PRINTING_DATASET,
                                    MODEL_BACKBONE,
                                    augment = 'val',
                                    split= 'val'
                                    )
    
    
    image, label = training_dataset[0]
    print( len(training_dataset))
    print(str(image.shape))
    # print(str(face))
    print(str(label))
    print(str(type(image)))
    print(str(type(label)))


    image, label = val_dataset[0]
    print( len(val_dataset))
    print(str(image.shape))
    # print(str(face))
    print(str(label))
    print(str(type(image)))
    print(str(type(label)))
    
    
    device = torch.device('cuda')
    # onnx runtime on fd onnx. 
    providers=["CUDAExecutionProvider"] # onnxrt cannot read cuda driver
    fd_session = ort.InferenceSession(PATH_TO_FD_MODEL,providers=providers)
    fas_model = ort.InferenceSession(PATH_TO_FAS_MODEL,providers=providers)
    print("   onnx cuda version:   "  +  (torch.version.cuda))