from libs.libs import *
from configs.config import *

class LivenessDetection():
    def __init__(self) -> None: 
        self.path_to_fas_model = PATH_TO_FAS_MODEL
        self.model_format =  FAS_FORMAT
        self.path_checkpoint = PATH_TO_CHECKPOINT_ONNX # test from checkpoint
        self.model = self.load_model()
        self.inference_device = INFERENCE_DEVICE
        
    # load onnx model  
    def load_model(self):
        '''  comment*
        
        
        '''
        if TEST_FROM_CHECKPOINT == True: 
            onnx_model = ort.InferenceSession(self.path_checkpoint) # path to check
            # CUDA
            if INFERENCE_DEVICE == "CUDA":
                onnx_model.set_providers(['CUDAExecutionProvider'])
            return onnx_model
        else:
            onnx_model = ort.InferenceSession(self.path_to_fas_model)
            if INFERENCE_DEVICE == "CUDA":
                onnx_model.set_providers(['CUDAExecutionProvider'])
            return onnx_model
        
    def pre_processing():
        '''  comment*
        
        
        '''
        pass
    # FAS inference on single image
    def infer_one_image(self, face):
        '''  comment*
        
        
        '''
        
        
        return 0
        
    def post_processing():
        '''  comment*
        
        
        '''
        pass
if __name__ == '__main__':
    obj_test = LivenessDetection()
