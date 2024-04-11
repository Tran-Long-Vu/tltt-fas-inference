from libs.libs import *
from configs.config import *


from src.scrfd import *
from src.image_dataset import ImageDataset
from src.liveness_detection import LivenessDetection
from configs.config import *


class FaceDetector():
    def __init__(self) -> None:
        '''  
        FaceDecector class. 
            loads scrfd model.
            if CUDA:
                use onnx.CUDAExecutionHandler
            Takes in a image_batch, stored in np array format:
            for img in image_batch, 
                preprocess,
                detect bounding box
                crop face bounding box
                post process for FAS inference.            
        Outputs a batch of post processed cropped faces, stored in np array format.
        '''  
        # self.data = self.load_data()
        self.path_to_fd_model = PATH_TO_FD_MODEL
        self.fas_model_backbone = FAS_BACKBONE
        self.model = self.load_model()
        # self.dataset = self.load_dataset()
        pass
    
    def load_model(self):
        '''
        Loads model from: /model/scrfd.onnx
            if CUDA, use onnx.CUDAExecutionHandler
        Output: scrfd onnx session.
        '''
        if self.path_to_fd_model is None:
            return 0
        scrfd = SCRFD(model_file=self.path_to_fd_model)
        scrfd.prepare(-1)
        if INFERENCE_DEVICE == "CUDA":
                scrfd.prepare(1)
        return scrfd

    def pre_processing(self, image_batch):
        '''  Input: an array of cv2 BGR image
        for image in batch
            reformat
            append
        Output: resized_batch 
        (bx3x640x640) formatted batch of np array image
        '''
        if len(image_batch) == 0:
            print('   no image batch')
            pass
        resized_batch = []
        for image in image_batch:
            # image = np.resize(image, (640,640,3)) # wrong format
            resized_batch.append(image)
            return resized_batch

    # detect single image
    def detect_one_batch(self,resized_batch): 
        '''  
        Input: resized image batch [[img1],[img2],[img3],...[]]
        for img in img_batch
            bbox = onnx.detect(img)
            # multiple bbox in one img: [[ [bbox1][bbox2]] ,
            # [ [bbox1][bbox2]],
            # [ [bbox1][bbox2]] 
            ...
            # ]
            batch.append
            return batch
        Output:  face_batch [ [f1 [b1,b2] ],[f2 [b1] ],[f3 [b1] ] ]
        containing bbox_batches
        '''
        if resized_batch is  None: 
            pass
        bbox_batch = []
        # there is something in resized.
        # resized_batch[0].shape()
        for image in resized_batch:
            
            bboxes, kpss = self.model.detect(resized_batch[0],0.5) # confidence threshold = 0.5 
            bbox_batch.append(bboxes) # contains all bboxes in one []
        # handle when FD cannot read any faces.
        # cannot read here.
        return bbox_batch
       

    # crop out face by bounding box
    def crop_bbox_batch(self,image_batch, bbox_batch, label_batch):
        '''  
        Input: a batch of bounding box of images [[bbox1],[bbox2],[bbox3],...]
        for index in batch
            cut image[i] with bbox[i]
            image_batch.append()
            bbox [a] , bbox[b] -> imagebatch[a] = label[a]
            link bbox with index label batch of that image.
        Output: batch of cropped faces [[face1],[face2],[face3],..]
        '''
        if image_batch is None or bbox_batch is None:
            pass
        cropped_batch = []
        cropped_label_batch = []
        for image, label in (zip(image_batch, label_batch)):
            for bboxes in bbox_batch:
                for bbox in bboxes:
                    x_min = int(bbox[0])
                    x_max = int(bbox[2])
                    y_min = int(bbox[1])
                    y_max = int(bbox[3])
                    
                    face = image[y_min:y_max, x_min:x_max]
                    cropped_batch.append(face)
                    cropped_label_batch.append(label)

        return cropped_batch, cropped_label_batch

    def post_processing(self, cropped_batch):
        '''  
        Input: an batch of cropped faces in nparray.
            for img in batch:
                resize(img) # depends on model backbone.
                batch.append(img)
        Output: face_batch
        a nparray batch of resized face, ready for FAS: [[face1][face2][face3],...]
        '''

        if len(cropped_batch) == 0: # check null
            pass
        face_batch = []
        for face in cropped_batch:
            # infer multiple face in one face.
            face = np.expand_dims(face,0)
            if self.fas_model_backbone == "mnv3":
                face = np.resize(face, (1,3,128,128))
            elif self.fas_model_backbone == "rn18":
                face = np.resize(face, (1,3,256,256))
            face_batch.append(face)
        return face_batch

    # todo: (later) inference realtime on single frame of video
    def live_detect_video(self, frame):
        '''  comment*
        
        
        '''
        

    # inference on batch image             
    def run_one_batch(self, image_batch, label_batch):
        '''  
        Input: batch of images & batch of labels
            for img in images
                run defs.
        Output: batch of cropped faces(images)
                batch of cropped labels relative to cropped faces.
        '''
        if len(image_batch) == 0:
            pass
        resized_batch = self.pre_processing(image_batch)
        bbox_batch = self.detect_one_batch(resized_batch)
        cropped_batch, cropped_label_batch = self.crop_bbox_batch(image_batch,bbox_batch, label_batch)
        face_batch = self.post_processing(cropped_batch)
        return face_batch, cropped_label_batch


if __name__ == '__main__':
    fd = FaceDetector()