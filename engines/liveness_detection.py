from libs import *
import os
from PIL import Image
import numpy as np
from configs.config import *

class LivenessDetection():
    def __init__(self) -> None: 
        self.path_to_fas_model = PATH_TO_FAS_MODEL
        self.model_format = "onnx"
        self.model = self.load_model()

    # load onnx model  
    def load_model(self):
        if self.model_format == 'onnx':
            import onnxruntime
            onnx_model = onnxruntime.InferenceSession(self.path_to_fas_model)
            print( 'Loaded:' + str(onnx_model._model_path))
            return onnx_model
        if self.model_format == 'pth': 
            import torch
            pth_model = torch.load(self.path_to_fas_model)
            pth_model.eval()
            print( 'Loaded: pth')
            return pth_model
        else:
            print("model error")
            return 0
        # TODO: if format == "jit"
    
    # FAS inference on single image
    def run_one_img_dir(self, face):
        if face is not None:
            ort_sess = self.model
            if self.path_to_fas_model == "./model/mnv3-fas.onnx":
                outputs = ort_sess.run(None, {'actual_input_1': face})
            elif self.path_to_fas_model == "./model/rn18-fas.onnx":
                outputs, x = ort_sess.run(None, {'input.1': face})
                return outputs
            else:
                print("  model  directory  error in configs")
        else:
            print("   FAS   cannot read face")
            return []
if __name__ == '__main__':
    obj_test = LivenessDetection()
