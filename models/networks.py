import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, x):
        return self.mlp.forward(x)

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self,ip_size):
        super().__init__()
        self.input_transform = Tnet(k=2)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(ip_size,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class ObsFeat2D(nn.Module):
    """Feature extractor for 1D organized point clouds"""

    def __init__(self, n_points, n_out=1024):
        super(ObsFeat2D, self).__init__()
        self.n_out = n_out
        k = 3
        p = int(np.floor(k / 2)) + 2
        self.conv1 = nn.Conv1d(2, 64, kernel_size=k, padding=p, dilation=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=k, padding=p, dilation=3)
        self.conv3 = nn.Conv1d(
            128, self.n_out, kernel_size=k, padding=p, dilation=3)
        self.mp = nn.MaxPool1d(n_points)
        # self.transform = Transform(2)

    def forward(self, x):
        assert(x.shape[1] == 2), "the input size must be <Bx2xL> "

        # x,_,_ = self.transform(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.mp(x)
        x = x.view(-1, self.n_out)  # <Bx1024>
        return x


class ObsFeatAVD(nn.Module):
    """Feature extractor for 2D organized point clouds"""
    def __init__(self, n_out=1024):
        super(ObsFeatAVD, self).__init__()
        self.n_out = n_out
        k = 3
        p = int(np.floor(k / 2)) + 2
        self.conv1 = nn.Conv2d(3,64,kernel_size=k,padding=p,dilation=3)
        self.conv2 = nn.Conv2d(64,128,kernel_size=k,padding=p,dilation=3)
        self.conv3 = nn.Conv2d(128,256,kernel_size=k,padding=p,dilation=3)
        self.conv4 = nn.Conv2d(256,self.n_out,kernel_size=k,padding=p,dilation=3)
        self.amp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        assert(x.shape[1]==3),"the input size must be <Bx3xHxW> "
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.amp(x) 
        x = x.view(-1,self.n_out) #<Bxn_out>
        return x


class LocNetReg2D(nn.Module):
    def __init__(self, n_points, out_dims):
        super(LocNetReg2D, self).__init__()
        self.obs_feat_extractor = ObsFeat2D(n_points)
        n_in = self.obs_feat_extractor.n_out
        self.fc = MLP([n_in, 512, 256, out_dims])

    def forward(self, obs):
        obs = obs.transpose(1, 2)
        obs_feat = self.obs_feat_extractor(obs)
        obs = obs.transpose(1, 2)

        x = self.fc(obs_feat)
        return x


class LocNetRegAVD(nn.Module):
    def __init__(self, out_dims):
        super(LocNetRegAVD, self).__init__()
        self.obs_feat_extractor = ObsFeatAVD()
        n_in = self.obs_feat_extractor.n_out
        self.fc = MLP([n_in, 512, 256, out_dims])

    def forward(self, obs):
        # obs: <BxHxWx3>
        bs = obs.shape[0]
        obs = obs.permute(0,3,1,2) # <Bx3xHxW>
        obs_feat = self.obs_feat_extractor(obs)
        obs = obs.permute(0,2,3,1)

        x = self.fc(obs_feat)
        return x
