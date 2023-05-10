import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from torch.nn import Conv2d
from torchvision import datasets,transforms
from einops import rearrange
from torch.optim import Adam as adam
from glob import glob
import random
import ipdb


class Gmodel(nn.Module):
    def __init__(self,backbone = 'tf_efficientnet_b4.ns_jft_in1k',fc=[256,3],):
        super().__init__()
        self.dla_model = timm.create_model(backbone,features_only=False,pretrained=True,num_classes=3)
        self.fc_layers = nn.ModuleList()
        fc_insize = 1000
        for i in range(len(fc)):
            self.fc_layers.append(nn.Linear(fc_insize,fc[i]))
            fc_insize=fc[i] 

    def forward(self,img):
        inputs = img
        features = self.dla_model(inputs)
        inputs = features
        # for i,layer in enumerate((self.fc_layers)):
        #     if i < len(self.fc_layers)-1:
        #         inputs = F.sigmoid(layer(inputs),)
        #     else:
        #         inputs =layer(inputs)

        return inputs


class Dataloader:
    def __init__(self,img_dir,batch_size):
        test_thresh = -500
        self.img_list =glob(img_dir)[:test_thresh]
        self.test_imgs = glob(img_dir)[test_thresh:]
        self.img_iter = 0
        self.batch_size = batch_size
        self.test_iter = 0
    
    def next_batch(self):

        curr_batch = self.img_list[self.img_iter:self.img_iter+self.batch_size]
        self.img_iter+=self.batch_size
        labels  = [ iname.split('_')[-2] for iname in curr_batch]
        imgs_np = np.array([cv2.resize(cv2.imread(im_name),(256,256)) for im_name in curr_batch],dtype=np.float32)/255.
        labels_encoded = []
        for label in labels:
            if label=='m':
                labels_encoded.append([1,0,0])
            elif label=='f':
                labels_encoded.append([0,1,0])
            elif label=='x':
                labels_encoded.append([0,0,1])
            else:
                import ipdb;ipdb.set_trace()
        labels_encoded= np.array(labels_encoded,dtype=np.float32)

        imgs_np = rearrange(imgs_np,'b h w c -> b c h w').astype(np.float32)
        return torch.tensor(imgs_np).to('cuda'),torch.tensor(labels_encoded).to('cuda')

    def batch_test(self):

        curr_batch = self.img_list[self.test_iter:self.test_iter+self.batch_size]
        labels  = [ iname.split('_')[-2] for iname in curr_batch]
        self.test_iter+=self.batch_size
        imgs_np = np.array([cv2.resize(cv2.imread(im_name),(256,256)) for im_name in curr_batch],dtype=np.float32)/255.
        labels_encoded = []
        for label in labels:
            if label=='m':
                labels_encoded.append([1,0,0])
            elif label=='f':
                labels_encoded.append([0,1,0])
            elif label=='x':
                labels_encoded.append([0,0,1])
            else:
                import ipdb;ipdb.set_trace()
        labels_encoded= np.array(labels_encoded,dtype=np.float32)

        imgs_np = rearrange(imgs_np,'b h w c -> b c h w').astype(np.float32)
        return torch.tensor(imgs_np).to('cuda'),torch.tensor(labels_encoded).to('cuda')       


    def on_epoch_ends(self):
        self.img_iter=0
        self.test_iter=0
        random.shuffle(self.img_list)



if __name__ =="__main__":

    model_path = 'model_checkpoints/fusion_10'
    model = Gmodel().to('cpu')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    imgs = []
    imgs.append(cv2.resize(cv2.imread('boy1.jpg'),(256,256))/255.)
    imgs.append(cv2.resize(cv2.imread('boy2.jpg'),(256,256))/255.)
    imgs.append(cv2.resize(cv2.imread('girl1.jpg'),(256,256))/255.)
    imgs.append(cv2.resize(cv2.imread('girl2.jpg'),(256,256))/255.)

    img = np.array(imgs)
    img = rearrange(img,'b h w c -> b c h w').astype(np.float32)
    predicted = model(torch.tensor(img).to('cpu'))
    print(predicted)

if __name__ =="__main1__":

    model = Gmodel().to('cuda')
    # model_path='model_checkpoints/fusion_50'
    # model.load_state_dict(torch.load(model_path))

    batch_size = 16
    MIX_W=1.3
    dl = Dataloader('gt_imgs/*',batch_size)
    number_of_batches = int(len(dl.img_list)/batch_size)
    EPOCHS=300
    loss = nn.CrossEntropyLoss()
    optim = adam(list(model.parameters()),lr=0.00001)
    for i in range(EPOCHS):
        model.train()
        loss_avg = 0
        avg_acc = 0
        avg_precision =[0,0,0]
        print('----------------------------------')
        for j in range(number_of_batches):
            imgs,labels = dl.next_batch()
            predicted = model(imgs)

            loss_val= loss(predicted,labels)
            predicted_idx = torch.argmax(predicted,dim=1)
            gt_idx = torch.argmax(labels,dim=1)
            accuracy = (torch.sum((predicted_idx==gt_idx)==True)/predicted_idx.shape[0]).to('cpu').detach()
            for k in range(3):
                avg_precision[k]+=sum([(predicted_idx[l]==k) and (gt_idx[l]==k) for l in range(len(predicted_idx))])/(sum(predicted_idx==k)+0.00001)
            
                
            avg_acc+=accuracy

            loss_val.backward()
            optim.step()
            loss_avg+=loss_val
        if i%10==0:
            torch.save(model.state_dict(),'model_checkpoints/fusion_'+str(i))

        labels_decoded = ['male','female','mix']
        if i%1==0:
            for k in range(3):
                print('the train AP of '+labels_decoded[k]+' is:',avg_precision[k]/number_of_batches)
        avg_acc/=number_of_batches
        loss_avg/=number_of_batches
        dl.on_epoch_ends()
        if i%3==0:
            print('In Epoch '+str(i)+' accuracy is:',avg_acc)
        avg_precision =[0,0,0]
        test_avg_acc=0
        with torch.no_grad():
            number_of_test_batches = int(len(dl.test_imgs)/batch_size)
            model.eval()
            for j in range(number_of_test_batches):
                imgs,labels = dl.batch_test()
                predicted = model(imgs)
                predicted_idx = torch.argmax(predicted,dim=1)
                gt_idx = torch.argmax(labels,dim=1)
                batch_accuracy = (torch.sum((predicted_idx==gt_idx)==True)/predicted_idx.shape[0]).to('cpu').detach()
                for k in range(3):
                    avg_precision[k]+=sum([(predicted_idx[l]==k) and (gt_idx[l]==k) for l in range(len(predicted_idx))])/(sum(predicted_idx==k)+0.000001)
                test_avg_acc+=batch_accuracy
            
        
        if i%1==0:
            for k in range(3):
                print('the test AP of '+labels_decoded[k]+' is:',avg_precision[k]/number_of_test_batches)
            print('test accuracy is:',test_avg_acc/number_of_test_batches)


        


        