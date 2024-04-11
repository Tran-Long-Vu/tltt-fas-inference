from libs.libs import *
from configs.config import *
from src.face_detector import FaceDetector
from src.liveness_detection import LivenessDetection
from src.image_dataset import ImageDataset
from src.scrfd import SCRFD
import onnxruntime as ort
class Checker():
    # can be replaced with pytest assert.
    def init():
        pass
    def check_cuda(self):
        if INFERENCE_DEVICE == "CUDA":
            print("    CUDA:  "  +  str(torch.cuda.is_available()))
            print("    CUDA device:  "  +  str(torch.cuda.get_device_name()))
            print("    CUDA NVCC: "  +  torch.version.cuda) 
        else:
            print("    Device: CPU    ")
        pass
        
    def check_data(self):
        dataset = ImageDataset(TEST_DATASET,
                    PATH_TO_PRINTING_TEST_DATASET,
                    FAS_BACKBONE,
                    augment = 'test',
                    split = 'test')
        return dataset
    
    def load_fas_model(self):
        fd = FaceDetector()
        return fd
    
    def load_fd_model(self):
        fas = LivenessDetection()
        pass
    
    def check_data2(self):
        pass
    
    def check_data3(self):
        pass
    
    def check_data4(self):
        pass



if __name__ == "__main__":
    checker = Checker()
    checker.check_cuda()
    
    dataset = checker.check_data()
    
    
    
    # get single sample: 
    image, label = dataset[0]
    image2, label2 = dataset[202]
    # label = label.cuda()
    # print(image.shape)
    # print("    label : " +  str(label))
    image_batch = [image,image2] # single inference
    label_batch = [label,label2]
    
    #  load fd model 
    fd = FaceDetector()
    face_batch, cropped_label_batch = fd.run_one_batch(image_batch, label_batch)

    # load fas model
