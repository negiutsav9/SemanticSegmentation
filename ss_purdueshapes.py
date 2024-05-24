'''################################################################################################
Filename: hw7_main.py
Author: Utsav Negi
Purpose: To implement multi-instance semantic segmentation on Purdue Shapes Dataset Objects.
################################################################################################'''
from DLStudio import *
import random
import os
import numpy as np
import torch
import torch.nn as nn
import gzip
import pickle
import matplotlib.pyplot as plt
import time
import copy
import torchvision
import pymsgbox

seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# Class used from DLStudio. Added new functions to support training and validation on models using different loss functions,
# create training loss plots and calculate Dice Loss
class SemanticSegmentation(nn.Module):             
        def __init__(self, dl_studio, max_num_objects, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
            super(SemanticSegmentation, self).__init__()
            self.dl_studio = dl_studio
            self.max_num_objects = max_num_objects
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class PurdueShapes5MultiObjectDataset(torch.utils.data.Dataset):
            def __init__(self, dl_studio, segmenter, train_or_test, dataset_file):
                super(SemanticSegmentation.PurdueShapes5MultiObjectDataset, self).__init__()
                max_num_objects = segmenter.max_num_objects
                if train_or_test == 'train' and dataset_file == "./data/PurdueShapes5MultiObject-10000-train.gz":
                    if os.path.exists("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt") and \
                              os.path.exists("torch_saved_PurdueShapes5MultiObject_label_map.pt"):
                        print("\nLoading training data from torch saved file")
                        self.dataset = torch.load("torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        self.label_map = torch.load("torch_saved_PurdueShapes5MultiObject_label_map.pt")
                        self.num_shapes = len(self.label_map)
                        self.image_size = dl_studio.image_size
                    else: 
                        print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                              """the dataset for this script. First time loading could take\n"""
                              """a few minutes.  Any subsequent attempts will only take\n"""
                              """a few seconds.\n\n\n""")
                        root_dir = dl_studio.dataroot
                        f = gzip.open(root_dir + dataset_file, 'rb')
                        dataset = f.read()
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                        torch.save(self.dataset, "torch_saved_PurdueShapes5MultiObject-10000_dataset.pt")
                        torch.save(self.label_map, "torch_saved_PurdueShapes5MultiObject_label_map.pt")
                        # reverse the key-value pairs in the label dictionary:
                        self.class_labels = dict(map(reversed, self.label_map.items()))
                        self.num_shapes = len(self.class_labels)
                        self.image_size = dl_studio.image_size
                else:
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.num_shapes = len(self.class_labels)
                    self.image_size = dl_studio.image_size

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image_size = self.image_size
                r = np.array( self.dataset[idx][0] )
                g = np.array( self.dataset[idx][1] )
                b = np.array( self.dataset[idx][2] )
                R,G,B = r.reshape(image_size[0],image_size[1]), g.reshape(image_size[0],image_size[1]), b.reshape(image_size[0],image_size[1])
                im_tensor = torch.zeros(3,image_size[0],image_size[1], dtype=torch.float)
                im_tensor[0,:,:] = torch.from_numpy(R)
                im_tensor[1,:,:] = torch.from_numpy(G)
                im_tensor[2,:,:] = torch.from_numpy(B)
                mask_array = np.array(self.dataset[idx][3])
                max_num_objects = len( mask_array[0] ) 
                mask_tensor = torch.from_numpy(mask_array)
                mask_val_to_bbox_map =  self.dataset[idx][4]
                max_bboxes_per_entry_in_map = max([ len(mask_val_to_bbox_map[key]) for key in mask_val_to_bbox_map ])
                ##  The first arg 5 is for the number of bboxes we are going to need. If all the
                ##  shapes are exactly the same, you are going to need five different bbox'es.
                ##  The second arg is the index reserved for each shape in a single bbox
                bbox_tensor = torch.zeros(max_num_objects,self.num_shapes,4, dtype=torch.float)
                for bbox_idx in range(max_bboxes_per_entry_in_map):
                    for key in mask_val_to_bbox_map:
                        if len(mask_val_to_bbox_map[key]) == 1:
                            if bbox_idx == 0:
                                bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                        elif len(mask_val_to_bbox_map[key]) > 1 and bbox_idx < len(mask_val_to_bbox_map[key]):
                            bbox_tensor[bbox_idx,key,:] = torch.from_numpy(np.array(mask_val_to_bbox_map[key][bbox_idx]))
                sample = {'image'        : im_tensor, 
                          'mask_tensor'  : mask_tensor,
                          'bbox_tensor'  : bbox_tensor }
                return sample

        def load_PurdueShapes5MultiObject_dataset(self, dataserver_train, dataserver_test ):   
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=4)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=4)


        class SkipBlockDN(nn.Module):
            def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
                super(SemanticSegmentation.SkipBlockDN, self).__init__()
                self.downsample = downsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convo1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.convo2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if downsample:
                    self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            def forward(self, x):
                identity = x                                     
                out = self.convo1(x)                              
                out = self.bn1(out)                              
                out = nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo2(out)                              
                    out = self.bn2(out)                              
                    out = nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out = out + identity
                    else:
                        out = out + torch.cat((identity, identity), dim=1) 
                return out


        class SkipBlockUP(nn.Module):
            def __init__(self, in_ch, out_ch, upsample=False, skip_connections=True):
                super(SemanticSegmentation.SkipBlockUP, self).__init__()
                self.upsample = upsample
                self.skip_connections = skip_connections
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.convoT1 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                self.convoT2 = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.bn2 = nn.BatchNorm2d(out_ch)
                if upsample:
                    self.upsampler = nn.ConvTranspose2d(in_ch, out_ch, 1, stride=2, dilation=2, output_padding=1, padding=0)
            def forward(self, x):
                identity = x                                     
                out = self.convoT1(x)                              
                out = self.bn1(out)                              
                out = nn.functional.relu(out)
                out  =  nn.ReLU(inplace=False)(out)            
                if self.in_ch == self.out_ch:
                    out = self.convoT2(out)                              
                    out = self.bn2(out)                              
                    out = nn.functional.relu(out)
                if self.upsample:
                    out = self.upsampler(out)
                    identity = self.upsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out = out + identity                              
                    else:
                        out = out + identity[:,self.out_ch:,:,:]
                return out
        

        class mUnet(nn.Module):
            def __init__(self, skip_connections=True, depth=16):
                super(SemanticSegmentation.mUnet, self).__init__()
                self.depth = depth // 2
                self.conv_in = nn.Conv2d(3, 64, 3, padding=1)
                ##  For the DN arm of the U:
                self.bn1DN  = nn.BatchNorm2d(64)
                self.bn2DN  = nn.BatchNorm2d(128)
                self.skip64DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(64, 64, skip_connections=skip_connections))
                self.skip64dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 64,   downsample=True, skip_connections=skip_connections)
                self.skip64to128DN = DLStudio.SemanticSegmentation.SkipBlockDN(64, 128, skip_connections=skip_connections )
                self.skip128DN_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128DN_arr.append(DLStudio.SemanticSegmentation.SkipBlockDN(128, 128, skip_connections=skip_connections))
                self.skip128dsDN = DLStudio.SemanticSegmentation.SkipBlockDN(128,128, downsample=True, skip_connections=skip_connections)
                ##  For the UP arm of the U:
                self.bn1UP  = nn.BatchNorm2d(128)
                self.bn2UP  = nn.BatchNorm2d(64)
                self.skip64UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip64UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, skip_connections=skip_connections))
                self.skip64usUP = DLStudio.SemanticSegmentation.SkipBlockUP(64, 64, upsample=True, skip_connections=skip_connections)
                self.skip128to64UP = DLStudio.SemanticSegmentation.SkipBlockUP(128, 64, skip_connections=skip_connections )
                self.skip128UP_arr = nn.ModuleList()
                for i in range(self.depth):
                    self.skip128UP_arr.append(DLStudio.SemanticSegmentation.SkipBlockUP(128, 128, skip_connections=skip_connections))
                self.skip128usUP = DLStudio.SemanticSegmentation.SkipBlockUP(128,128, upsample=True, skip_connections=skip_connections)
                self.conv_out = nn.ConvTranspose2d(64, 5, 3, stride=2,dilation=2,output_padding=1,padding=2)

            def forward(self, x):
                ##  Going down to the bottom of the U:
                x = nn.MaxPool2d(2,2)(nn.functional.relu(self.conv_in(x)))          
                for i,skip64 in enumerate(self.skip64DN_arr[:self.depth//4]):
                    x = skip64(x)                
        
                num_channels_to_save1 = x.shape[1] // 2
                save_for_upside_1 = x[:,:num_channels_to_save1,:,:].clone()
                x = self.skip64dsDN(x)
                for i,skip64 in enumerate(self.skip64DN_arr[self.depth//4:]):
                    x = skip64(x)                
                x = self.bn1DN(x)
                num_channels_to_save2 = x.shape[1] // 2
                save_for_upside_2 = x[:,:num_channels_to_save2,:,:].clone()
                x = self.skip64to128DN(x)
                for i,skip128 in enumerate(self.skip128DN_arr[:self.depth//4]):
                    x = skip128(x)                
        
                x = self.bn2DN(x)
                num_channels_to_save3 = x.shape[1] // 2
                save_for_upside_3 = x[:,:num_channels_to_save3,:,:].clone()
                for i,skip128 in enumerate(self.skip128DN_arr[self.depth//4:]):
                    x = skip128(x)                
                x = self.skip128dsDN(x)
                ## Coming up from the bottom of U on the other side:
                x = self.skip128usUP(x)          
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x[:,:num_channels_to_save3,:,:] =  save_for_upside_3
                x = self.bn1UP(x)
                for i,skip128 in enumerate(self.skip128UP_arr[:self.depth//4]):
                    x = skip128(x)                
                x = self.skip128to64UP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[self.depth//4:]):
                    x = skip64(x)                
                x[:,:num_channels_to_save2,:,:] =  save_for_upside_2
                x = self.bn2UP(x)
                x = self.skip64usUP(x)
                for i,skip64 in enumerate(self.skip64UP_arr[:self.depth//4]):
                    x = skip64(x)                
                x[:,:num_channels_to_save1,:,:] =  save_for_upside_1
                x = self.conv_out(x)
                return x

        def dice_loss(self, preds: torch.Tensor , ground_truth: torch.Tensor , epsilon=1e-6):
            preds = torch.flatten(preds)
            ground_truth = torch.flatten(ground_truth)
            numerator = torch.sum(torch.dot(preds, ground_truth))
            denominator = torch.dot(preds,preds) + torch.dot(ground_truth, ground_truth)
            dice_coeffecient = (2*numerator + epsilon) /(denominator + epsilon)
            return (1 - dice_coeffecient)

        def training_for_semantic_segmentation_mse(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion_mse = nn.MSELoss()
            optimizer = torch.optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            loss = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_segmentation = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    im_tensor   = im_tensor.to(self.dl_studio.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.dl_studio.device)                 
                    bbox_tensor = bbox_tensor.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    output = net(im_tensor) 
                    segmentation_loss = criterion_mse(output, mask_tensor)
                    segmentation_loss.backward()
                    optimizer.step()
                    running_loss_segmentation += segmentation_loss.item()  
                    if i%500==499:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        avg_loss_segmentation = running_loss_segmentation / float(500)
                        print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   MSE loss: %.3f" % (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_segmentation))
                        FILE1.write("%.3f\n" % avg_loss_segmentation)
                        FILE1.flush()
                        running_loss_segmentation = 0.0
                        loss.append(avg_loss_segmentation)  
            print("\nFinished Training\n")
            plt.plot(loss)
            plt.title('Training Loss with MSE')
            plt.ylabel('Segmentation Loss')
            plt.xlabel('Iterations')
            plt.savefig('../../training_loss_mse.png')
            torch.save(net.state_dict(), "./net_mse.pth")
            return loss
        
        def training_for_semantic_segmentation_dice(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion_mse = nn.MSELoss()
            optimizer = torch.optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            loss = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_segmentation = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    im_tensor   = im_tensor.to(self.dl_studio.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.dl_studio.device)                 
                    bbox_tensor = bbox_tensor.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    output = net(im_tensor) 
                    segmentation_loss = self.dice_loss(output, mask_tensor)
                    segmentation_loss.backward()
                    optimizer.step()
                    running_loss_segmentation += segmentation_loss.item()  
                    if i%500==499:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        avg_loss_segmentation = running_loss_segmentation / float(500)
                        print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   DICE loss: %.3f" % (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_segmentation))
                        FILE1.write("%.3f\n" % avg_loss_segmentation)
                        FILE1.flush()
                        running_loss_segmentation = 0.0
                        loss.append(avg_loss_segmentation)  
            print("\nFinished Training\n")
            plt.plot(loss)
            plt.title('Training Loss with DICE')
            plt.ylabel('Segmentation Loss')
            plt.xlabel('Iterations')
            plt.savefig('../../training_loss_dice.png')
            torch.save(net.state_dict(), "./net_dice.pth")
            return loss
        
        def training_for_semantic_segmentation_mse_dice(self, net):        
            filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE1 = open(filename_for_out1, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            criterion_mse = nn.MSELoss()
            optimizer = torch.optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            loss = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss_segmentation = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    im_tensor   = im_tensor.to(self.dl_studio.device)
                    mask_tensor = mask_tensor.type(torch.FloatTensor)
                    mask_tensor = mask_tensor.to(self.dl_studio.device)                 
                    bbox_tensor = bbox_tensor.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    output = net(im_tensor) 
                    segmentation_loss = criterion_mse(output, mask_tensor) + (20*self.dice_loss(output, mask_tensor))
                    segmentation_loss.backward()
                    optimizer.step()
                    running_loss_segmentation += segmentation_loss.item()  
                    if i%500==499:    
                        current_time = time.perf_counter()
                        elapsed_time = current_time - start_time
                        avg_loss_segmentation = running_loss_segmentation / float(500)
                        print("[epoch=%d/%d, iter=%4d  elapsed_time=%3d secs]   MSE+DICE loss: %.3f" % (epoch+1, self.dl_studio.epochs, i+1, elapsed_time, avg_loss_segmentation))
                        FILE1.write("%.3f\n" % avg_loss_segmentation)
                        FILE1.flush()
                        running_loss_segmentation = 0.0
                        loss.append(avg_loss_segmentation)  
            print("\nFinished Training\n")
            plt.plot(loss)
            plt.title('Training Loss with MSE+DICE')
            plt.ylabel('Segmentation Loss')
            plt.xlabel('Iterations')
            plt.savefig('../../training_loss_mse_dice.png')
            torch.save(net.state_dict(), "./net_mse_dice.pth")
            return loss


        def run_code_for_testing_semantic_segmentation(self, net, net_path):
            net.load_state_dict(torch.load(net_path))
            batch_size = self.dl_studio.batch_size
            image_size = self.dl_studio.image_size
            max_num_objects = self.max_num_objects
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    im_tensor,mask_tensor,bbox_tensor =data['image'],data['mask_tensor'],data['bbox_tensor']
                    if i % 50 == 0:
                        print("\n\n\n\nShowing output for test batch %d: " % (i+1))
                        outputs = net(im_tensor)                        
                        ## In the statement below: 1st arg for batch items, 2nd for channels, 3rd and 4th for image size
                        output_bw_tensor = torch.zeros(batch_size,1,image_size[0],image_size[1], dtype=float)
                        for image_idx in range(batch_size):
                            for layer_idx in range(max_num_objects): 
                                for m in range(image_size[0]):
                                    for n in range(image_size[1]):
                                        output_bw_tensor[image_idx,0,m,n]  =  torch.max( outputs[image_idx,:,m,n] )
                        display_tensor = torch.zeros(7 * batch_size,3,image_size[0],image_size[1], dtype=float)
                        for idx in range(batch_size):
                            for bbox_idx in range(max_num_objects):   
                                bb_tensor = bbox_tensor[idx,bbox_idx]
                                for k in range(max_num_objects):
                                    i1 = int(bb_tensor[k][1])
                                    i2 = int(bb_tensor[k][3])
                                    j1 = int(bb_tensor[k][0])
                                    j2 = int(bb_tensor[k][2])
                                    output_bw_tensor[idx,0,i1:i2,j1] = 255
                                    output_bw_tensor[idx,0,i1:i2,j2] = 255
                                    output_bw_tensor[idx,0,i1,j1:j2] = 255
                                    output_bw_tensor[idx,0,i2,j1:j2] = 255
                                    im_tensor[idx,0,i1:i2,j1] = 255
                                    im_tensor[idx,0,i1:i2,j2] = 255
                                    im_tensor[idx,0,i1,j1:j2] = 255
                                    im_tensor[idx,0,i2,j1:j2] = 255
                        display_tensor[:batch_size,:,:,:] = output_bw_tensor
                        display_tensor[batch_size:2*batch_size,:,:,:] = im_tensor

                        for batch_im_idx in range(batch_size):
                            for mask_layer_idx in range(max_num_objects):
                                for i in range(image_size[0]):
                                    for j in range(image_size[1]):
                                        if mask_layer_idx == 0:
                                            if 25 < outputs[batch_im_idx,mask_layer_idx,i,j] < 85:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 1:
                                            if 65 < outputs[batch_im_idx,mask_layer_idx,i,j] < 135:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 2:
                                            if 115 < outputs[batch_im_idx,mask_layer_idx,i,j] < 185:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 3:
                                            if 165 < outputs[batch_im_idx,mask_layer_idx,i,j] < 230:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50
                                        elif mask_layer_idx == 4:
                                            if outputs[batch_im_idx,mask_layer_idx,i,j] > 210:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 255
                                            else:
                                                outputs[batch_im_idx,mask_layer_idx,i,j] = 50

                                display_tensor[2*batch_size+batch_size*mask_layer_idx+batch_im_idx,:,:,:]= outputs[batch_im_idx,mask_layer_idx,:,:]
                        self.dl_studio.display_tensor_as_image(
                           torchvision.utils.make_grid(display_tensor, nrow=batch_size, normalize=True, padding=2, pad_value=10))

def plot_loss(mse_losses, dice_losses, mse_dice_losses):

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(10, 10),
        gridspec_kw={
            "wspace": 0.75,
            "hspace": 0.75,
        },
    )
    fig.tight_layout()
    ax1.title.set_text("MSE Loss")
    ax1.plot(mse_losses)
    ax2.title.set_text("DICE Loss")
    ax2.plot(dice_losses)
    ax3.title.set_text("MSE + DICE Loss")
    ax3.plot(mse_dice_losses)
    ax1.set_xlabel("Iterations")
    ax2.set_xlabel("Iterations")
    ax3.set_xlabel("Iterations")
    ax1.grid()
    ax2.grid()
    ax3.grid()

    plt.figure()
    plt.title("MSE Loss, DICE Loss and MSE+DICE Loss")
    plt.plot(mse_losses, label="MSE Loss")
    plt.plot(dice_losses, label="DICE Loss")
    plt.plot(mse_dice_losses, label="MSE+DICE Loss")
    plt.xlabel("Iterations")
    plt.legend()

    plt.show()

# main ML pipeline. Similar to semantic_segmentation.py
if __name__=='__main__':
	dls = DLStudio(
#       dataroot = "/home/kak/ImageDatasets/PurdueShapes5MultiObject/",
		dataroot = "./data/",
        image_size = [64,64],
        path_saved_model = "./saved_model",
        momentum = 0.9,
        learning_rate = 1e-4,
        epochs = 6,
        batch_size = 4,
        classes = ('rectangle','triangle','disk','oval','star'),
        use_gpu = True,
    )
	segmenter = SemanticSegmentation( dl_studio = dls, max_num_objects = 5)
	dataserver_train = SemanticSegmentation.PurdueShapes5MultiObjectDataset(
                          train_or_test = 'train',
                          dl_studio = dls,
                          segmenter = segmenter,
                          dataset_file = "PurdueShapes5MultiObject-10000-train.gz", 
                        )
	dataserver_test = DLStudio.SemanticSegmentation.PurdueShapes5MultiObjectDataset(
                          train_or_test = 'test',
                          dl_studio = dls,
                          segmenter = segmenter,
                          dataset_file = "PurdueShapes5MultiObject-1000-test.gz"
                        )
	segmenter.dataserver_train = dataserver_train
	segmenter.dataserver_test = dataserver_test

	segmenter.load_PurdueShapes5MultiObject_dataset(dataserver_train, dataserver_test)

	model = segmenter.mUnet(skip_connections=True, depth=16)

	number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("\n\nThe number of learnable parameters in the model: %d\n" % number_of_learnable_params)

	num_layers = len(list(model.parameters()))
	print("\nThe number of layers in the model: %d\n\n" % num_layers)

	mse_loss = segmenter.training_for_semantic_segmentation_mse(model)
	dice_loss = segmenter.training_for_semantic_segmentation_dice(model)
	mse_dice_loss = segmenter.training_for_semantic_segmentation_mse_dice(model)
	
	plot_loss(mse_loss, dice_loss, mse_dice_loss)
 
	segmenter.run_code_for_testing_semantic_segmentation(model, "./net_mse.pth")
	segmenter.run_code_for_testing_semantic_segmentation(model, "./net_dice.pth")
	segmenter.run_code_for_testing_semantic_segmentation(model, "./net_mse_dice.pth")
