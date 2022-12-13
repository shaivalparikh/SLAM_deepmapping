from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from .networks import LocNetReg2D, LocNetRegAVD, MLP
from utils import transform_to_global_2D, transform_to_global_AVD

LIDAR_TO_WORLD_EULER = (np.pi/2,-np.pi/2,0)
THRESH = 0.0002

def rotation_from_euler_zyx(alpha, beta, gamma):
    R = np.zeros((3, 3), dtype='double')
    
    R[0, 0] = np.cos(alpha) * np.cos(beta)
    R[0, 1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    R[0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    
    R[1, 0] = np.sin(alpha) * np.cos(beta)
    R[1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    R[1, 2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    
    R[2, 0] = -np.sin(beta)
    R[2, 1] =  np.cos(beta) * np.sin(gamma)
    R[2, 2] =  np.cos(beta) * np.cos(gamma)
    
    return R

def lidar_to_world(arr: np.ndarray):
	'''
	just a rotation, so done separately
	'''
	if arr.shape[1] != 3:
		raise ValueError(f'arr should have 3 columns, instead got {arr.shape[1]}')

	T = rotation_from_euler_zyx(*LIDAR_TO_WORLD_EULER)
	return (T @ arr.T).T

def make_homogenous_and_transform(arr: np.ndarray, T: np.ndarray):
	if arr.shape[1] != 3 or T.shape != (3, 4):
		raise ValueError(f'arr should have 3 columns and T should be 3x4, instead got {arr.shape[1]} and {T.shape}')

	T_4 = np.vstack((T, [0, 0, 0, 1]))
	arr_4 = np.c_[arr, np.ones(arr.shape[0])]

	transformed_arr = T_4 @ arr_4.T
	return transformed_arr.T[:, :-1]

def pcd_to_occupancy(pcd):
    # if pcd.shape[1] != 3:
    #     raise ValueError(f'pcd should have 3 columns, instead got {pcd.shape[1]}')\
    pcd = 1000*pcd
    # print("PCD:",(torch.unique(pcd)).shape)
    pcd = pcd.round()
    # print("Unique:",(torch.unique(pcd)).shape)
    x_min = pcd[:, 0].min()
    x_max = pcd[:, 0].max()

    z_min = pcd[:, 1].min()
    z_max = pcd[:, 1].max()

    occupancy = torch.zeros((int(x_max - x_min + 1), int(z_max - z_min + 1)))

    for i in range(pcd.shape[0]):
        x = int(pcd[i, 0] - x_min)
        z = int(pcd[i, 1] - z_min)

        occupancy[x, z] += 1
    # print("Extreme Vals: ",x_min,x_max,z_min,z_max)
    
    occupancy /= pcd.shape[0]
    
    # print(np.count_nonzero(occupancy > THRESH))
    return occupancy > THRESH

def occupancy_generation(init_est):
    # print(init_est.shape)
    output = torch.zeros((init_est.shape[0],5120,1))
    # print("OP shape:",output.shape)
    for i in range(init_est.shape[0]):

      occup = pcd_to_occupancy(init_est[i])
      # print(occup.shape)

      occup = torch.flatten(occup)
      # print(occup.shape)
      occup.sort(descending = True)
      # print(occup.shape)
      occup = occup[:5120]
      # print(occup.shape)
      occup = torch.reshape(occup,(1,5120,1))
      # print(occup.shape)
      output[i] = occup

      # print(occup.shape)

    return output






def get_M_net_inputs_labels(occupied_points, unoccupited_points):
    """
    get global coord (occupied and unoccupied) and corresponding labels
    """
    n_pos = occupied_points.shape[1]
    inputs = torch.cat((occupied_points, unoccupited_points), 1)
    bs, N, _ = inputs.shape

    gt = torch.zeros([bs, N, 1], device=occupied_points.device)
    gt.requires_grad_(False)
    gt[:, :n_pos, :] = 1
    return inputs, gt


def sample_unoccupied_point(local_point_cloud, n_samples, center):
    """
    sample unoccupied points along rays in local point cloud
    local_point_cloud: <BxLxk>
    n_samples: number of samples on each ray
    center: location of sensor <Bx1xk>
    """
    bs, L, k = local_point_cloud.shape
    center = center.expand(-1,L,-1) # <BxLxk>
    unoccupied = torch.zeros(bs, L * n_samples, k,
                             device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        unoccupied[:, (idx - 1) * L:idx * L, :] = center + (local_point_cloud-center) * fac
    return unoccupied

class DeepMapping2D(nn.Module):
    def __init__(self, loss_fn, n_obs=256, n_samples=19, dim=[2, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping2D, self).__init__()
        self.n_obs = n_obs
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetReg2D(n_points=n_obs, out_dims=3)
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points,sensor_pose):
        # obs_local: <BxLx2>
        # sensor_pose: init pose <Bx1x3>
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points

        self.pose_est = self.loc_net(self.obs_local)
        # print(self.pose_est)
        self.obs_global_est = transform_to_global_2D(
            self.pose_est, self.obs_local)
        # print(self.obs_global_est.shape)
        if self.training:
            sensor_center = sensor_pose[:,:,:2]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_2D(
                self.pose_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            # self.occp_prob = self.occup_net(inputs)
            # print(self.occp_prob.shape)
            self.occp_prob = occupancy_generation(self.obs_global_est)
            # print(test.shape)
            self.occp_prob = self.occp_prob.to("cuda")
            loss = self.compute_loss()
            return loss


    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=4, gamma=0.1)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss



class DeepMapping_AVD(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, loss_fn, n_samples=35, dim=[3, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping_AVD, self).__init__()
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetRegAVD(out_dims=3) # <x,z,theta> y=0
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points,sensor_pose):
        # obs_local: <BxHxWx3> 
        # valid_points: <BxHxW>
        
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points
        self.pose_est = self.loc_net(self.obs_local)

        bs = obs_local.shape[0]
        self.obs_local = self.obs_local.view(bs,-1,3)
        self.valid_points = self.valid_points.view(bs,-1)
        
        self.obs_global_est = transform_to_global_AVD(
            self.pose_est, self.obs_local)

        if self.training:
            sensor_center = sensor_pose[:,:,:2]
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples,sensor_center)
            self.unoccupied_global = transform_to_global_AVD(
                self.pose_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            self.occp_prob = self.occup_net(inputs)
            loss = self.compute_loss()
            return loss

    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=2, gamma=0.9)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss

