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
                    MODEL_BACKBONE,
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
    # cuda
    if INFERENCE_DEVICE == "CUDA":
        image = image.cuda()
        label = label.cuda()
        print(image.shape)
        print(label.shape)
    
    print("    image loading done")
    
    
    fd = FaceDetector()    
    # if CUDA: 
    fd.model.session.set_providers(['CUDAExecutionProvider']) # set provider for SCRFD

    # test face detector first. input: nparray or tensor image. Output: array of np array : [[face1],[face2]] or array of tensor: [[tensor1, tensor2]]
   
    
    faces = fd.detect_one_face(image) # use when cpu.
    print(faces) # nparray
    # todo: inference within GPU debugging:
    
    # output = fas.detect_one_face(face)
    # print(output)
    # print(label)