from configs.config import *
from libs import *
from engines.scrfd import SCRFD
from data_script.image_dataset import ImageDataset
from configs.config import *
import mlflow


# todo - configs file
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
experiment = mlflow.set_experiment(" VHT Fine Tuning RN18 ")
run_id = experiment.experiment_id
class FasTrainer():
    def __init__(self) -> None:
        self.model_backbone = "rn18" # change between 'rn18' and 'mnv3'
        self.attack_type = ATTACK_TYPE
        self.model = self.load_model()
        self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr = 1e-5, # 0.0001
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_ckp_dir = PATH_TO_SAVE_CHECKPOINT
        self.epochs = NO_EPOCHS
        
    # init image dataset format
    def load_train_image_dataset(self):
        training_dataset =  ImageDataset(       TRAIN_DATASET,
                                                PATH_TO_PRINTING_DATASET,
                                                MODEL_BACKBONE,
                                                augment = 'train',
                                                split= 'train'
                                                )
        return training_dataset
    def load_val_image_dataset(self):
        dataset = ImageDataset(     VAL_DATASET,
                                    PATH_TO_PRINTING_DATASET,
                                    MODEL_BACKBONE,
                                    augment = 'val',
                                    split= 'val'
                                    )
        return dataset
    
    # init video dataset format # later
    def load_video_dataset(self):
        return 0
    
    # Dataloader
    def load_image_dataloader(self, dataset):
        dataloader = DataLoader(dataset,
                                batch_size=BATCH_SIZE, 
                                shuffle=True,
                                num_workers=4)
        return dataloader
        # Dataloader
    def load_video_dataloader(self):
        return 0
    
    # model onnx2torch
    def load_model(self):
        from DGFAS import DG_model
        model = DG_model("resnet18")
        if TRAIN_FROM_CHECKPOINT == True:
            # if checkpoint
            model.load_state_dict(torch.load(PATH_TO_CHECKPOINT_MODEL)) #error
            print("    loaded checkpoint  " )
            return model
        else:
            return model        
    
    # run printing attack dataset
    def train_printing_attack(self,
                              train_loader,
                              num_epochs,
                              model,
                              optimizer,
                              device,
                              save_ckp_dir        
        ):
        print("\nStart Training ...\n" + " = "*16)
        # delete all contents in log.
        model = model.to(device)
        print( "    Device:    "+ str(device))
        best_accuracy = 0.0
        # Configure logger
        logging.basicConfig(filename='./training_logs/training.log', level=logging.INFO)
        
        params = {"epochs": NO_EPOCHS, 
                    "optimizer": self.optimizer,
                    "device": self.device,
                    "checkpoint": "first",
                    "batch_size": BATCH_SIZE,
                    "backbone" : MODEL_BACKBONE,
                    "dataset" : PATH_TO_TRAIN_DATASET,
                    "attack type": ATTACK_TYPE
                    }
        
        epoches =[]
        train_accuracies=[]
        train_losses=[]
        train_fars=[]
        train_frrs=[]
        train_hters=[]
        

        val_accuracies=[]
        val_losses=[]
        val_fars=[]
        val_frrs=[]
        val_hters=[]
        with mlflow.start_run( run_name = 'epoch 5, cvpr set on resnet18, printing'):
            for epoch in range(1, num_epochs + 1):
                
                train_TP = 0.01
                train_TN = 0.01
                train_FP = 0.01
                train_FN = 0.01
                
                val_TP = 0.01
                val_TN = 0.01
                val_FP = 0.01
                val_FN = 0.01
                
                epoches.append(epoch)
                
                print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)

                model.train()
                running_loss, running_corrects,  = 0.0, 0.0, 
                for images, labels in tqdm.tqdm(train_loader): 
                
                    images, labels = images.to(device), labels.to(device)
                    
                    logits = model(images.float()) 
                    logits = torch.cat(logits, dim=1) 
                    loss = F.cross_entropy(logits, labels) 

                    predicted_labels = torch.argmax(logits, dim=1)

                    train_TP += torch.sum((predicted_labels == 1) & (labels == 1)).item()
                    train_TN += torch.sum((predicted_labels == 0) & (labels == 0)).item()
                    train_FP += torch.sum((predicted_labels == 1) & (labels == 0)).item()
                    train_FN += torch.sum((predicted_labels == 0) & (labels == 1)).item()
                    loss.backward()
                    optimizer.step(), optimizer.zero_grad()

                    running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
                
                    train_loss, train_accuracy,  = running_loss/len(train_loader.dataset), running_corrects/len(train_loader.dataset), 

                
                train_FAR = train_FP / (train_FP + train_TN)
                train_FRR = train_FN / (train_FN + train_TP)
                train_HTER = (train_FAR + train_FRR) / 2

                
                print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
                    "train", 
                    train_loss, train_accuracy, 
                ))
                
                
                

                # Log the training metrics
                logging.info("epoch: {}".format(epoch))
                logging.info("train_TP: {}".format(train_TP))
                logging.info("train_TN: {}".format(train_TN))
                logging.info("train_FP: {}".format(train_FP))
                logging.info("train_FN: {}".format(train_FN))
                logging.info("train_FAR: {}".format(train_FAR))
                logging.info("train_FRR: {}".format(train_FRR))
                logging.info("train_HTER: {}".format(train_HTER))
                logging.info("--------------")
                
                train_accuracies.append(train_accuracy)
                train_losses.append(train_loss)
                train_fars.append(train_FAR)
                train_frrs.append(train_FRR)
                train_hters.append(train_HTER)

                
                with torch.no_grad():
                    model.eval()
                    running_loss, running_corrects,  = 0.0, 0.0, 
                    for images, labels in tqdm.tqdm(train_loader):
                        images, labels = images.to(device), labels.to(device)
                        logits = model(images.float())
                        logits = torch.cat(logits, dim=1) # concat dim = 1. prevent format error
                        predicted_labels = torch.argmax(logits, dim=1)

                        val_TP += torch.sum((predicted_labels == 1) & (labels == 1)).item()
                        val_TN += torch.sum((predicted_labels == 0) & (labels == 0)).item()
                        val_FP += torch.sum((predicted_labels == 1) & (labels == 0)).item()
                        val_FN += torch.sum((predicted_labels == 0) & (labels == 1)).item()
                        loss = F.cross_entropy(logits, labels)

                        running_loss, running_corrects,  = running_loss + loss.item()*images.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels.data).item(), 
                val_loss, val_accuracy,  = running_loss/len(train_loader.dataset), running_corrects/len(train_loader.dataset), 
                # wandb.log({"val_loss":val_loss, "val_accuracy":val_accuracy, }, step = epoch)


                val_FAR = val_FP / (val_FP + val_TN)
                val_FRR = val_FN / (val_FN + val_TP)
                val_HTER = (val_FAR + val_FRR) / 2

                

                
                print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
                    "val", 
                    val_loss, val_accuracy, 
                ))
                # Log the training metrics
                logging.info("epoch: {}".format(epoch))
                logging.info("val_TP: {}".format(val_TP))
                logging.info("val_TN: {}".format(val_TN))
                logging.info("val_FP: {}".format(val_FP))
                logging.info("val_FN: {}".format(val_FN))
                logging.info("val_FAR: {}".format(val_FAR))
                logging.info("val_FRR: {}".format(val_FRR))
                logging.info("val_HTER: {}".format(val_HTER))
                logging.info("--------------")
                
                # logging for plotting
                val_accuracies.append(val_accuracy)
                val_losses.append(val_loss)
                val_fars.append(val_FAR)
                val_frrs.append(val_FRR)
                val_hters.append(val_HTER)
                
                # Log the hyperparameters
                mlflow.log_params(params)
                mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                mlflow.log_metric("train_far", train_FAR, step=epoch)
                mlflow.log_metric("val_far", val_FAR, step=epoch)

                mlflow.log_metric("train_frr", train_FRR, step=epoch)
                mlflow.log_metric("val_frr", val_FRR, step=epoch)
                                            
                mlflow.log_metric("train_hter", train_HTER, step=epoch)
                mlflow.log_metric("val_hter", val_HTER, step=epoch) 
                # save checkpoint
                if val_accuracy > best_accuracy:
                    if not os.path.exists("checkpoints/"):
                        os.makedirs("checkpoints/")
                    torch.save(model.state_dict(), "{}/fas-best.pth".format(save_ckp_dir))
                    
                    # to onnx.
                    dummy_input = torch.randn(1, 3, 256, 256).to(device)
                    torch.onnx.export(model, dummy_input, "./checkpoints/fas-best.onnx")
                    print("onnx checkpoint saved.")
                    best_accuracy = val_accuracy
            
        print("\nFinish Training ...\n" + " = "*16)

        return {
            "train_loss":train_loss, "train_accuracy":train_accuracy, 
            "val_loss":val_loss, "val_accuracy":val_accuracy, 
        }

    def train_replay_attack(self):
        
        pass
    
    def visualize():
        pass

if __name__ == '__main__':
    fas_trainer = FasTrainer()
    train_set = fas_trainer.load_train_image_dataset()
    val_set = fas_trainer.load_val_image_dataset()

    train_loader = fas_trainer.load_image_dataloader(train_set)
    val_loader = fas_trainer.load_image_dataloader(val_set)
    # if training attack
    fas_trainer.train_printing_attack(train_loader,
                                      fas_trainer.epochs,
                                      fas_trainer.model,
                                      fas_trainer.optimizer,
                                      fas_trainer.device,
                                      fas_trainer.save_ckp_dir)
    # if replay attack:
    # fas_trainer.run_replay_atk()
    pass
