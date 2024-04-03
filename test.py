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
class FasSolution():
    def __init__(self) -> None:
        
        self.fas_model_backbone = "rn18" # change between 'rn18' and 'mnv3'
        
        self.path_to_labeled_data = PATH_TO_PRINTING_TEST_DATASET
        self.path_to_video_dir = PATH_TO_REPLAY_TEST_DATA

        self.model_format = "onnx"
        self.printing_dataset = self.load_printing_dataset()
        self.replay_dataset = self.load_video_dataset
        
        self.video_dataset = self.load_video_dataset()
        self.provider = ""
        self.cuda = torch.cuda.is_available()

        if self.cuda == True:
            print("cuda ready")
            self.device = torch.device("cuda")
        else:
            print("cpu ready")
            self.device = torch.device("cpu")
        if self.device == torch.device("cpu"):
            self.provider = ["CPUExecutionProvider"]
        elif self.device == torch.device("cuda"):
            self.provider = ["CUDAExecutionProvider"]
        else: 
            print("device issue")

        if self.fas_model_backbone == "mnv3":
            self.fas = LivenessDetection()
            self.fd = FaceDetector()
            self.fd.fas_model_backbone = "mnv3"
            
            
        elif self.fas_model_backbone == "rn18":
            self.fas = LivenessDetection()
            self.fd = FaceDetector()
            self.fd.fas_model_backbone = "rn18"
            
        else:
            print ("incorrect model")
        pass
    
    # init image dataset format
    def load_printing_dataset(self):
        dataset = ImageDataset(TEST_DATASET,
                    PATH_TO_PRINTING_TEST_DATASET,
                    MODEL_BACKBONE,
                    augment = 'test',
                    split = 'test'
                                    )
        # print(   '    len  ds     '  +   str(len(dataset)))
        return dataset
    
    # init video dataset format
    def load_video_dataset(self):
        video_dataset = VideoDataset(self.path_to_video_dir)
        return video_dataset
    
    # Dataloader
    def load_dataloader(self):
        dataset = self.load_dataset()
        print("dir:" + self.path_to_labeled_data)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = 1, 
        )
        print("loaded torch.dataloader " +
              " | model format:  " + str(self.model_format) +
              " |  Batches: " + str(len(loader)))
        return loader

    # run printing attack dataset
    def run_on_image_dataset(self):
        print("Start Testing ...\n" + " = "*16  + "  with backbone: " +  str(self.fd.fas_model_backbone))
        inference_times = []
        true_labels = []
        predicted_labels = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for image, label in tqdm.tqdm(self.printing_dataset):
            if self.printing_dataset is not None:
                start_time = time.time()
                formatted_faces = self.fd.run_on_img_dir(image)
                if formatted_faces is not None:
                    outputs = self.fas.run_one_img_dir(formatted_faces)
                    if self.fas_model_backbone == 'mnv3':
                        prediction = outputs[0][0]
                        if prediction is not None:
                            if label == 0:
                                logit = round(prediction[0])
                            elif label == 1:
                                logit = round(prediction[1])
                                
                            if logit == 1 and label == 1:
                                TP += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 0 and label == 0:
                                TN += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 1 and label == 0:
                                FP += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 0 and label == 1:
                                FN += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                        else:
                            continue
                    elif self.fas_model_backbone == 'rn18':
                        prediction = outputs[0] 
                        prediction_softmax = np.exp(prediction) / np.sum(np.exp(prediction))
                        # print("   softmax prediction array:    "   +    str(prediction_softmax))
                        
                        if prediction_softmax is not None:
                            if label == 0:
                                logit = round(prediction_softmax[0])
                            elif label == 1:
                                logit = round(prediction_softmax[1])
                                
                            if logit == 1 and label == 1:
                                TP += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 0 and label == 0:
                                TN += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 1 and label == 0:
                                FP += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                            elif logit == 0 and label == 1:
                                FN += 1
                                true_labels.append(label)
                                predicted_labels.append(logit)
                        else:
                            continue
                else:
                    formatted_faces = []
                    continue
                end_time = time.time()
                inference_time = end_time - start_time
                inference_times.append(inference_time)
            else:
                print("no image loaded")
                return 0
        average_time = sum(inference_times) / len(self.printing_dataset)
        print(metrics.classification_report(true_labels, predicted_labels, digits=4))
        far =  FP / (FP + TN)  * 100
        frr = FN / (FN + TP) * 100
        print("FAR: " + "{:.2f}".format(far) +  "%")
        print("FRR: " + "{:.2f}".format(frr) +  "%")
        print("HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        print("\nFinish Testing ...\n" + " = "*16)
        print("average inference time for one image: " +  "{:.2f}".format(average_time)+ " seconds.")
        print("total inference time: " +  "{:.2f}".format(sum(inference_times)) + " seconds.")
    pass

    # run single video by path
    def run_on_video_file(self, path_to_test_video):
        
        video_frames = []
        widths = []
        heights = []
        score_0s = []
        score_1s = []
        final_preds = []
        img_paths = []
        ground_truths = []
        
        total_time = 0
        frame_count = 0
        video = cv2.VideoCapture(path_to_test_video)
        start_time = time.time()
        
        if not video.isOpened():
            print("video not found.")
            return 0
        
        while video.isOpened():
            frame_count += 1
            print( "         frame  count :    " + str(frame_count))
            video_frames.append(frame_count) 
            ret, frame = video.read()
            if not ret:
                    break
            if frame is not None:
                result  = self.fd.run_and_record_frame_video(frame)
                if result is not None:
                    face, width, height = result
                    print( "         W & H :   "   +   str( width)   +   "  &  "   +  str(height))
                    widths.append(width) 
                    heights.append(height) 
                    outputs = self.fas.run_one_img_dir(face)
                    if self.fas_model_backbone == 'mnv3':
                        prediction = outputs[0][0]
                        score_0 = prediction[0]
                        score_1 = prediction[1]
                        print('      score 0:      ' + str(score_0) ) 
                        print('      score 1:      ' + str(score_1) )
                        score_0s.append(score_0)
                        score_1s.append(score_1)
                        if score_0 > score_1:
                            final_pred = 0
                            final_preds.append(final_pred)
                            print('         final predict:   ' +  str(final_pred) + "\n")
                        elif score_0 <= score_1:
                            final_pred = 1
                            final_preds.append(final_pred)
                            print('         final predict:      ' +  str(final_pred) + "\n")
                    elif self.fas_model_backbone == 'rn18': 
                        prediction = outputs[0]
                        prediction_softmax = np.exp(prediction) / np.sum(np.exp(prediction))
                        score_0 = prediction_softmax[0]
                        score_1 = prediction_softmax[1]
                        print('      score 0:      ' + str(score_0) ) 
                        print('      score 1:      ' + str(score_1) )
                        score_0s.append(score_0)
                        score_1s.append(score_1)
                        if score_0 > score_1:
                            final_pred = 0
                            final_preds.append(final_pred)
                            print('        final predict:      ' +  str(final_pred)  + "\n")
                        elif score_0 <= score_1:
                            final_pred = 1
                            final_preds.append(final_pred)
                            print('        final predict:      ' +  str(final_pred) + "\n")
                    image_name = f'image_{frame_count}.png'
                    if not os.path.exists("frames/"):
                        os.makedirs("frames/")
                    path_to_save_img = ('frames/' + image_name)
                    
                    cv2.imwrite(path_to_save_img, face)
                    img_paths.append(path_to_save_img)
                    ground_truth = int(os.path.basename(os.path.dirname(path_to_test_video)))
                    ground_truths.append(ground_truth)
                else:
                    width = 0
                    height = 0
                    score_0 = 0.0
                    score_1 = 0.0
                    final_pred = 2
                    ground_truth = int(os.path.basename(os.path.dirname(path_to_test_video)))
                    
                    widths.append(width)
                    heights.append(height)
                    score_0s.append(score_0)
                    score_1s.append(score_0)
                    final_preds.append(final_pred)
                    img_paths.append('')
                    ground_truths.append(ground_truth)
                    
        video.release()
        end_time = time.time()
        total_time = ((end_time - start_time))
        print("    total inference time for video: "  +  str( total_time ) + " seconds ")
        print("     total number of frames " + str(video_frames[-1]))
        print(" average width : " + str( sum(widths) / len(widths) ))
        print(" average height : " + str( sum(heights) / len(heights)))
        print(" average clean score : " + str(sum(score_0s) / len(score_0s)))
        print(" average spoof score : " + str(sum(score_1s) / len(score_1s)))
        video_frames.pop(-1)
        df = pd.DataFrame({
             'video_frames': video_frames,
             'widths': widths,
             'heights': heights,
             'score_0s': score_0s,
             'score_1s': score_1s,
             'final_preds': final_preds,
             'img_paths': img_paths,
             'ground_truths': ground_truths
         })
        print(df)
        filename = 'real_video_benchmark.csv'
        df.to_csv('results/' +  filename, index=False)
        pass
   
    # run on replay attack
    def run_on_video_dataset(self):
        print("Start Testing ...\n" + " = "*16  + "  with backbone: " +  str(self.fas_model_backbone))
        inference_times = []
        video_TP = 0.01
        video_TN = 0.01
        video_FP = 0.01
        video_FN = 0.01
        predicted_labels = []
        true_labels = []
        for frame_array, label in tqdm.tqdm(self.video_dataset):
            if self.video_dataset is not None: 
                start_time = time.time()
                img_TP = 0.0001
                img_TN = 0.0001
                img_FP = 0.0001
                img_FN = 0.0001
                for frame in frame_array:
                    if frame is not None:
                        formatted_faces = self.fd.run_on_img_dir(frame)
                        if formatted_faces is not None:
                            outputs = self.fas.run_one_img_dir(formatted_faces)
                            if self.fas_model_backbone == 'mnv3':
                                prediction = outputs[0][0]
                                if prediction is not None:
                                    if label == 0:
                                        logit = round(prediction[0])
                                    elif label == 1:
                                        logit = round(prediction[1])
                                    if logit == 1 and label == 1:
                                        img_TP += 1
                                    elif logit == 0 and label == 0:
                                        img_TN += 1
                                    elif logit == 1 and label == 0:
                                        img_FP += 1
                                    elif logit == 0 and label == 1:
                                        img_FN += 1
                                else:
                                    continue
                            elif self.fas_model_backbone == 'rn18': 
                                prediction = outputs[0] 
                                prediction_softmax = np.exp(prediction) / np.sum(np.exp(prediction))
                                if prediction_softmax is not None:
                                    if label == 0:
                                        logit = round(prediction_softmax[0])
                                    elif label == 1:
                                        logit = round(prediction_softmax[1])
                                    if logit == 1 and label == 1:
                                        img_TP += 1
                                    elif logit == 0 and label == 0:
                                        img_TN += 1
                                    elif logit == 1 and label == 0:
                                        img_FP += 1
                                    elif logit == 0 and label == 1:
                                        img_FN += 1
                                else:
                                    continue
                        else:
                            formatted_faces = []
                            continue
            label_0_ratio = (img_TN / img_FP)
            label_1_ratio = (img_TP / img_FN)
            if label_1_ratio > 1.00 and label == 1:
                video_prediction = 1
            elif label_1_ratio <= 1.00 and label == 1:
                video_prediction = 0
            elif label_0_ratio >= 1.00 and label == 0: 
                video_prediction = 0
            elif label_0_ratio < 1.00 and label == 0: 
                video_prediction = 1
            if video_prediction == 1 and label == 1:
                video_TP += 1.0
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            elif video_prediction == 0 and label == 0:
                video_TN += 1.0
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            elif video_prediction == 1 and label == 0:
                video_FP += 1.0
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            elif video_prediction == 0 and label == 1:
                video_FN += 1.0
                predicted_labels.append(video_prediction)
                true_labels.append(label)
            
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
        average_time = sum(inference_times) /  len(self.video_dataset)
        far =  video_FP / (video_FP + video_TN)  * 100
        frr = video_FN / (video_FN + video_TP) * 100
        print(metrics.classification_report(true_labels, predicted_labels, digits=4))
        
        print("     FAR: " + "{:.2f}".format(far) +  "%")
        print("     FRR: " + "{:.2f}".format(frr) +  "%")
        print("     HTER: " +  "{:.2f}".format((far + frr)/2) +  "%" )
        print("average inference time for one video: " +  "{:.2f}".format(average_time)+ " seconds.")
        print("total inference time all video: " +  "{:.2f}".format(sum(inference_times) / 60) + " minutes.")
        print("\nFinish Testing ...\n" + " = "*16)
        pass
    
# Uncomment to test other attacks.
if __name__ == '__main__':
    
    # error in imagre dataset code
    
    fas_solution = FasSolution()
    # configs: inference type: 'printing', 'replay', 'live'
    if INFERENCE_TYPE == 'printing':
        fas_solution.run_on_image_dataset()
    elif INFERENCE_TYPE == 'replay':
        fas_solution.run_on_video_dataset()
    elif INFERENCE_TYPE == 'live':
        fas_solution.run_on_video_file(PATH_TO_SINGLE_VIDEO)
    pass