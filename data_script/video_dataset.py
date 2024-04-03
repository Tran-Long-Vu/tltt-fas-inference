# import
from libs import *
from PIL import Image# Class init
class VideoDataset(torch.utils.data.Dataset):
    # Init
    def __init__(self, path_to_video_dir, ) -> None: 
        self.all_video_paths = glob.glob(path_to_video_dir + "*/*")
        # self.transform = tf.ToTensor()
        
    def transform_t(self,
                       img):
        t_image = self.transform(img)
        
        return t_image

    # Length
    def __len__(self, 
    ):
        return len(self.all_video_paths)

    # Get label
    def get_label(self, video_path):
        label = int(os.path.basename(os.path.dirname(video_path)))
        return label
    
    # get image tensor
    def __getitem__(self, index ):
        video_frames = []
        total_time = 0
        video_path = self.all_video_paths[index]
        label = self.get_label(video_path)
        count = 0
        # cv2
        video = cv2.VideoCapture(video_path)
        start_time = time.time()
        if not video.isOpened():
            print("video not found.")
            return[]
        
        while video.isOpened():
            count += 1
            
            ret, frame = video.read()
            video_frames.append(frame)
            # print("added frame" )
            # print("frame: " + str(frame.shape))
            # print("type: " + str(type(frame)))
            # print("frame count: " + str(count))
            if not ret:
                break
            
        video.release()
        end_time = time.time()
        total_time = ((end_time - start_time))
        # t_image = self.transform_t(video)
        # t_label = torch.tensor(label)
        
        # print(image.size(1))
        average_time = (total_time) / len(self)
        # print("total frame count:  "  +  str(count))
        # print("average frame load time: " + "{:.2f}".format(average_time))
        # print("total time to load a single video: " +  "{:.2f}".format((total_time)) )
        return video_frames, label