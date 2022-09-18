from glob import glob
from os import listdir
from os.path import splitext
import torch.utils.data
from pytorchtools import EarlyStopping
import cmapy
from functions import *
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
time1 = time.time()
data_folder = "/data/max/auyu's code/dataset"
#data_path = "{}/processedThebe".format(data_folder)
data_path = "{}".format(data_folder)
model_path='./checkpoints/table7_pidinet.pth'
best_model_fpath = 'pidinet.pth'
best_iou_threshold = 0.5
epoches = 50#!
patience = 25
batch_size = 64
modelNo = "pidi"
if modelNo == "unet":
    from model_zoo.UNET import Unet
    model = Unet()
    print("use model Unet")
elif modelNo == "deeplab":
    from model_zoo.DEEPLAB.deeplab import DeepLab
    model = DeepLab(backbone='xception', num_classes=1, output_stride=16)
    print("use model DeepLab")
elif modelNo == "hed":
    from model_zoo.HED import HED
    model = HED()
    print("use model HED")
elif modelNo == "rcf":
    from model_zoo.RCF import RCF
    model = RCF()
    print("use model RCF")
elif modelNo == "pidi":
    from model_zoo.PIDI import *
    model = pidinet(config='carv4', dil=True, sa=True)
    
        
    print("use model pidinet")
else:
    print("please select a valid model")
model = model.cuda()
summary(model, (1, 96, 96))
if model_path != '':
    #------------------------------------------------------#
    #   权值文件请看README
    #------------------------------------------------------#
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items(
    # ) if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

class faultsDataset(torch.utils.data.Dataset):
    #     def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):

    def __init__(self, imgs_dir, masks_dir):
        #         self.train = train
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(
            imgs_dir) if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask = np.load("{}/{}.npy".format(self.masks_dir, idx))
        img = np.load("{}/{}.npy".format(self.images_dir, idx))
        #         mask_file = glob(self.masks_dir + idx + '.npy')
        #         img_file = glob(self.images_dir + idx + '.npy')

        #         assert len(mask_file) == 1, \
        #             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        #         assert len(img_file) == 1, \
        #             f'Either no image or multiple images found for the ID {idx}: {img_file}'
        #         mask = np.load(mask_file[0])
        #         img = np.load(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (img, mask)


# In[32]:


faults_dataset_train = faultsDataset(imgs_dir="{}/train1/seismic".format(data_path),
                                     masks_dir="{}/train1/annotation".format(data_path))
faults_dataset_val = faultsDataset(imgs_dir="{}/val/seismic".format(data_path),
                                   masks_dir="{}/val/annotation".format(data_path))

#batch_size = 64#!swin_96_48_seed_test.pth

train_loader = torch.utils.data.DataLoader(dataset=faults_dataset_train,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val,
                                         batch_size=batch_size,
                                         shuffle=False)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9,weight_decay=0.0002)

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.0002)
if modelNo == "hed" or modelNo == "rcf" or modelNo == "pidi":
    print("optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9, weight_decay=0.00002)")
    #print('optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)')
    #print("optimizer = torch.optim.Adam(model.parameters(),lr=0.05 betas=(0.9, 0.99))")
if modelNo == "unet" or modelNo == "deeplab":
    print("optimizer = torch.optim.Adam(model.parameters(), lr=0.01)")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', factor=0.1, patience=5, verbose=True)
bceloss = nn.BCELoss()
mean_train_losses = []
mean_val_losses = []
mean_train_accuracies = []
mean_val_accuracies = []
t_start = time.time()
early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0)
for epoch in range(epoches):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    labelled_val_accuracies = []

    model.train()
    for _,(images, masks) in enumerate(tqdm(train_loader)):
        torch.cuda.empty_cache()
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        outputs = model(images)
        # print(outputs.shape)

        loss = torch.zeros(1).cuda()
        y_preds = outputs
        if modelNo == "unet" or modelNo == "deeplab":
            loss = bceloss(outputs, masks)
        elif modelNo == "hed":
            for o in range(5):
                loss = loss + cross_entropy_loss_HED(outputs[o], masks)
            loss = loss + bceloss(outputs[-1], masks)
            y_preds = outputs[-1]
        elif modelNo == "rcf":
            for o in outputs:
                loss = loss + cross_entropy_loss_RCF(o, masks)
            y_preds = outputs[-1]
        elif modelNo == "pidi":
            if not isinstance(outputs, list):
                loss = cross_entropy_loss_RCF(outputs, masks)
            else:
                loss = 0
                for o in outputs:
                    loss += cross_entropy_loss_RCF(o, masks)
                y_preds = outputs[-1]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        train_acc = iou_pytorch(predicted_mask.squeeze(
            1).byte(), masks.squeeze(1).byte())
        train_accuracies.append(train_acc.mean())

    model.eval()
    for _,(images, masks) in enumerate(tqdm(val_loader)):
        torch.cuda.empty_cache()
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        outputs = model(images)

        loss = torch.zeros(1).cuda()
        y_preds = outputs
        if modelNo == "unet" or modelNo == "deeplab":
            loss = bceloss(outputs, masks)
        elif modelNo == "hed":
            for o in range(5):
                loss = loss + cross_entropy_loss_HED(outputs[o], masks)
            loss = loss + bceloss(outputs[-1], masks)
            y_preds = outputs[-1]
        elif modelNo == "rcf":
            for o in outputs:
                loss = loss + cross_entropy_loss_RCF(o, masks)
            y_preds = outputs[-1]
        elif modelNo == "pidi":
            if not isinstance(outputs, list):
                loss = cross_entropy_loss_RCF(outputs, masks)
            else:
                loss = 0
                for o in outputs:
                    loss += cross_entropy_loss_RCF(o, masks)
                y_preds = outputs[-1]
        val_losses.append(loss.data)
        predicted_mask = y_preds > best_iou_threshold
        val_acc = iou_pytorch(predicted_mask.byte(), masks.squeeze(1).byte())
        val_accuracies.append(val_acc.mean())

    mean_train_losses.append(torch.mean(torch.stack(train_losses)))
    mean_val_losses.append(torch.mean(torch.stack(val_losses)))
    mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
    mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies))) 

    scheduler.step(torch.mean(torch.stack(val_losses)))
    early_stopping(torch.mean(torch.stack(val_losses)),
                   model, best_model_fpath)
    #torch.save(model.state_dict(),'./pidimodel/pidi_val_')
    if early_stopping.early_stop:
        print("Early stopping")
        break


    torch.cuda.empty_cache()

    for param_group in optimizer.param_groups:
        learningRate = param_group['lr']

    # Print Epoch results
    t_end = time.time()

    print('Epoch: {}. Train Loss: {}. Val Loss: {}. Train IoU: {}. Val IoU: {}. Time: {}. LR: {}'
          .format(epoch + 1, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)),
                  torch.mean(torch.stack(train_accuracies)), torch.mean(
                      torch.stack(val_accuracies)), t_end - t_start,
                  learningRate))
    torch.save(model.state_dict(),'./pidi_model/pidi_epoch{}_val{}.pth'.format(epoch,torch.mean(torch.stack(val_accuracies))))
    print('/n')
    print('model has been saved in dire ')
    t_start = time.time()
mean_train_losses = np.asarray(torch.stack(mean_train_losses).cpu())
mean_val_losses = np.asarray(torch.stack(mean_val_losses).cpu())
mean_train_accuracies = np.asarray(torch.stack(mean_train_accuracies).cpu())
mean_val_accuracies = np.asarray(torch.stack(mean_val_accuracies).cpu())

fig = plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# train_loss_series = pd.Series(mean_train_losses)
# val_loss_series = pd.Series(mean_val_losses)
# train_loss_series.plot(label="train_loss")
# val_loss_series.plot(label="validation_loss")
# plt.legend()
plt.subplot(1,1,1)
train_acc_series = pd.Series(mean_train_accuracies)
val_acc_series = pd.Series(mean_val_accuracies)
np.savetxt("训练准确率1.txt", train_acc_series)
np.savetxt("验证准确率1.txt", val_acc_series)
train_acc_series.plot(label="train_acc")
val_acc_series.plot(label="validation_acc")
plt.grid(linestyle='--')
plt.tick_params(direction='in')
plt.legend()
plt.savefig('./pidi_model/pidi_attention_acc.png')

totaltime = time.time()-time1
print("total cost {} hours".format(totaltime/3600)) 
