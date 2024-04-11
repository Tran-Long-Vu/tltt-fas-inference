import __main__
from configs.config import *
# import
from libs.libs import *

import torch
class ImageDataset(torch.utils.data.Dataset):
    # Init
    def __init__(self,
                 dataset_name,
                 path_to_data,
                 model_backbone,
                 augment,
                 split,
    ) -> None: 
        
        self.dataset_name = dataset_name
        self.model_backbone = model_backbone
        self.augment = augment
        self.split = split
        if self.dataset_name == "CVPR23":
            img_extension = '.jpg'
            text_extension = '.txt'
            self.all_image_paths = []
            self.all_face_paths = []
            for root, dirs, files in os.walk(path_to_data):
                for file in files:
                    if file.endswith(img_extension):
                        image_name = os.path.splitext(file)[0]
                        txt_file = image_name + text_extension
                        txt_path = os.path.join(root, txt_file)
                        if os.path.exists(txt_path):
                            self.all_image_paths.append(os.path.join(root, file))
                            self.all_face_paths.append(txt_path)
        elif self.dataset_name == "HAND_CRAWL":
            self.all_image_paths = glob.glob(path_to_data + "*/*.jpg")
        # splitting ds
        if self.split == "train":
                self.all_image_paths = self.all_image_paths[:int(0.8 * len(self.all_image_paths))]
        elif self.split == "val":
            self.all_image_paths = self.all_image_paths[int(0.8 * len(self.all_image_paths)):]
        elif self.split == 'test':
            pass
            
        
        if augment == 'train':
            self.transform = tf.Compose([
                            tf.ToTensor(),
                            #tf.Resize((256, 256)),
                            tf.RandomHorizontalFlip(),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            tf.ColorJitter(brightness=0.2, contrast=0.2),
                            tf.RandomRotation(30),
                        ])
        elif augment == 'val' or INFERENCE_DEVICE == "CUDA":
            self.transform = tf.Compose([
                                tf.ToTensor(),
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                #tf.Resize((256, 256)),
                            ])
     #length
    def __len__(self, 
    ):
        return len(self.all_image_paths)

    # Get label
    def get_label(self, image_path):
        if self.dataset_name == "CVPR23":
            image_path = image_path.replace("\\", "/") # to linux
            label = (image_path.split("/")[-3])  
            if label == 'living':
                label = 0
                return label
            if label != 'spoof':
                label = (image_path.split("/")[-4])  # move to spoof
                if label == "spoof":
                    label = 1
                    return label
        elif self.dataset_name == "HAND_CRAWL":
            label = int(os.path.basename(os.path.dirname(image_path)))
            return label
    
    # get image tensor
    def __getitem__(self, index ):
        if self.augment == 'train':
            image_path = self.all_image_paths[index]
            face_path = self.all_face_paths[index]
            
            image = cv2.imread(image_path)
            label = self.get_label(image_path)
            # cut image
            with open(face_path, 'r') as file:
                bbox = [next(file).strip() for _ in range(2)]
            x1, y1 = map(int, bbox[0].split())
            x2, y2 = map(int, bbox[1].split())    

            image = cv2.imread(image_path)
            cropped_image = image[y1:y2, x1:x2] 
            label = self.get_label(image_path)
            t_label = torch.tensor(label)
            t_image = self.transform(cropped_image)
            return t_image, t_label   
        
        elif self.augment == 'val':
            image_path = self.all_image_paths[index]
            face_path = self.all_face_paths[index]
            
            image = cv2.imread(image_path)
            label = self.get_label(image_path)
            
            t_image = self.transform(image)
            t_label = torch.tensor(label)
            return t_image, t_label

        elif self.augment == 'test':
            image_path = self.all_image_paths[index] # all image paths
            image = cv2.imread(image_path)
            image = np.array(image).astype(np.float32)
            label = self.get_label(image_path)
            return image, label
        # if self.model_backbone == 'mnv3':





