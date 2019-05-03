import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork2(torch.nn.Module):
    
    def __init__(self):
        
        super(SiameseNetwork2, self).__init__()
        
        self.convolution_layer_3_64   = torch.nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 9, stride = 2, padding = 4)
        self.convolution_layer_64_64  = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.convolution_layer_64_128 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.convolution_layer_128_256 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 1, stride = 1, padding = 0)
        self.convolution_layer_256_384 = torch.nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 1, stride = 1, padding = 0)
        self.convolution_layer_384_512 = torch.nn.Conv2d(in_channels = 384, out_channels = 512, kernel_size = 1, stride = 1, padding = 0)
        
        
        self.relu = torch.nn.ReLU()
        self.maxpooling_layer = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.global_maxpooling_layer = torch.nn.MaxPool2d(kernel_size = 6)
        
        self.batchNorm_layer_64       = torch.nn.BatchNorm2d(64)
        self.batchNorm_layer_96       = torch.nn.BatchNorm2d(96)
        self.batchNorm_layer_128      = torch.nn.BatchNorm2d(128)
        self.batchNorm_layer_256      = torch.nn.BatchNorm2d(256)
        self.batchNorm_layer_384      = torch.nn.BatchNorm2d(384)
        self.batchNorm_layer_512      = torch.nn.BatchNorm2d(512)

        
        ###############################################
        ## Convolution Operations for the Sub_Blocks ##
        ###############################################
        self.convolution_layer_128_64 = torch.nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 1, stride = 1, padding = 0)
        self.convolution_layer_64_64  = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.convolution_layer_64_128 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)

        self.convolution_layer_256_64 = torch.nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 1, stride = 1, padding = 0)
        self.convolution_layer_64_64  = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.convolution_layer_64_256 = torch.nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = 1, stride = 1, padding = 0)
        
        self.convolution_layer_384_96 = torch.nn.Conv2d(in_channels = 384, out_channels = 96, kernel_size = 1, stride = 1, padding = 0)
        self.convolution_layer_96_96  = torch.nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 1)
        self.convolution_layer_96_384 = torch.nn.Conv2d(in_channels = 96, out_channels = 384, kernel_size = 1, stride = 1, padding = 0)
       
        self.convolution_layer_512_128 = torch.nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.convolution_layer_128_128 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.convolution_layer_128_512 = torch.nn.Conv2d(in_channels = 128, out_channels = 512, kernel_size = 1, stride = 1, padding = 0)
        

    def Sub_Block(self, x, filter_size):
        
        data_channel_depth = x.size(1)
        
        if(filter_size==64 and data_channel_depth==128):
            
            x_save = x
            x = self.batchNorm_layer_128(x)
            x = self.convolution_layer_128_64(x)
            x = self.relu(x)

            x = self.batchNorm_layer_64(x)
            x = self.convolution_layer_64_64(x)
            x = self.relu(x)

            x = self.batchNorm_layer_64(x)
            x = self.convolution_layer_64_128(x)
            output = torch.add(x_save, x)
            output = self.relu(output)

        if(filter_size==64 and data_channel_depth==256):
            
            x_save = x
            x = self.batchNorm_layer_256(x)
            x = self.convolution_layer_256_64(x)
            x = self.relu(x)
            
            x = self.batchNorm_layer_64(x)
            x = self.convolution_layer_64_64(x)
            x = self.relu(x)

            x = self.batchNorm_layer_64(x)
            x = self.convolution_layer_64_256(x)
            output = torch.add(x_save, x)
            output = self.relu(output)
            
        if(filter_size==96 and data_channel_depth==384):
            
            x_save = x
            x = self.batchNorm_layer_384(x)
            x = self.convolution_layer_384_96(x)
            x = self.relu(x)
            
            x = self.batchNorm_layer_96(x)
            x = self.convolution_layer_96_96(x)
            x = self.relu(x)

            x = self.batchNorm_layer_96(x)
            x = self.convolution_layer_96_384(x)
            output = torch.add(x_save, x)
            output = self.relu(output)

        if(filter_size==128 and data_channel_depth==512):
            
            x_save = x
            x = self.batchNorm_layer_512(x)
            x = self.convolution_layer_512_128(x)
            x = self.relu(x)

            x = self.batchNorm_layer_128(x)
            x = self.convolution_layer_128_128(x)
            x = self.relu(x)

            x = self.batchNorm_layer_128(x)
            x = self.convolution_layer_128_512(x)
            output = torch.add(x_save, x)
            output = self.relu(output)

        return output
        
        
    def forward_once(self, x):

        x = self.convolution_layer_3_64(x)   # 192X192X64
        x = self.relu(x)
        x = self.maxpooling_layer(x)         # 96X96X64
        x = self.batchNorm_layer_64(x)
        x = self.convolution_layer_64_64(x)  # 96X96X64
        x = self.relu(x)
        x = self.batchNorm_layer_64(x)
        x = self.convolution_layer_64_64(x)  # 96X96X64
        x = self.relu(x)
        
        x = self.maxpooling_layer(x)         # 48X48X64
        x = self.batchNorm_layer_64(x)
        x = self.convolution_layer_64_128(x) # 48X48X128
        
        # Sub-Block 
        for _ in range(4):                   # 48X48X128
            x = self.Sub_Block(x, 64)

        x = self.maxpooling_layer(x)         # 24X24X128
        x = self.batchNorm_layer_128(x)
        x = self.convolution_layer_128_256(x)# 24X24X256
        
        # Sub-Block 
        for _ in range(4):                   # 24X24X256
            x = self.Sub_Block(x, 64)
            
        x = self.maxpooling_layer(x)         # 12X12X256        
        x = self.batchNorm_layer_256(x)
        x = self.convolution_layer_256_384(x)# 12X12X384

        # Sub-Block 
        for _ in range(4):                   # 12X12X384
            x = self.Sub_Block(x, 96)
        
        x = self.maxpooling_layer(x)         # 6X6X384        
        x = self.batchNorm_layer_384(x)
        x = self.convolution_layer_384_512(x)# 6X6X512

        # Sub-Block 
        for _ in range(4):                   # 6X6X512
            x = self.Sub_Block(x, 128)

        x = self.global_maxpooling_layer(x)
        
        x = x.view(-1,512)

        x = F.normalize(x, p=2, dim=1)
        
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
