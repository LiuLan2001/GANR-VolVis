import numpy as np
import torch
from torch import nn

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features):
        super(ResBlock,self).__init__()
        self.sine1 = SineLayer(in_features, out_features)
        self.sine2 = SineLayer(out_features, out_features)
        self.flag = (in_features!=out_features)
        if self.flag:
            self.transform = SineLayer(in_features,out_features)
    
    def forward(self,features):
        outputs = self.sine1(features)
        outputs = self.sine2(outputs)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)

class CoordNet(nn.Module):
    def __init__(self, in_features, out_features, init_features=64,num_res = 10):
        super(CoordNet,self).__init__()

        self.num_res = num_res

        self.net = []

        self.net.append(ResBlock(in_features,init_features))
        self.net.append(ResBlock(init_features,2*init_features))
        self.net.append(ResBlock(2*init_features,4*init_features))

        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))
        self.net.append(ResBlock(4*init_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features) 
        
    def forward(self, input):
        return self.linear(input)

class BottleNeckBlock(nn.Module):
    def __init__(self,in_features):
        super(BottleNeckBlock,self).__init__()
        self.net = []

        self.net.append(SineLayer(in_features, in_features//4))

        self.net.append(SineLayer(in_features//4, in_features//4))

        self.net.append(SineLayer(in_features//4, in_features))

        self.net = nn.Sequential(*self.net)
    
    def forward(self, features):
        outputs = self.net(features)
        return outputs+features
    
class CoordNetBottleNeck(nn.Module):
    def __init__(self, in_features, out_features, init_features,num_res):
        super(CoordNetBottleNeck,self).__init__()

        self.num_res = num_res

        self.net = []
        self.net.append(SineLayer(in_features,init_features))
        self.net.append(SineLayer(init_features,2*init_features))
        self.net.append(SineLayer(2*init_features,4*init_features))

        for i in range(self.num_res):
            self.net.append(BottleNeckBlock(4*init_features))
        self.net.append(SineLayer(4*init_features,out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        out = self.net(coords)
        return out
    

class Grid(nn.Module):
    """
        input x needs to be within [0, 1]
    """
    def __init__(self,
                 feature_dim: int,
                 grid_dim: int,
                 num_lvl: int,
                 max_res: int,
                 min_res: int,
                 hashtable_power: int,
                 force_cpu: bool
                 ):
        super().__init__()
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = 'cuda:0'
        self.feature_dim = feature_dim
        self.grid_dim = grid_dim
        self.num_lvl = num_lvl
        self.max_res = max_res
        self.min_res = min_res
        self.hashtable_power = hashtable_power
        self.prime = [3367900313, 2654435761, 805459861]
        self.max_entry = 2 ** self.hashtable_power
        self.factor_b = np.exp((np.log(self.max_res) - np.log(self.min_res)) / (self.num_lvl - 1))

        self.resolutions = []
        for i in range(self.num_lvl):
            self.resolutions.append(np.floor(self.min_res * (self.factor_b**i)))

        self.hashtable = nn.ParameterList([])
        for res in self.resolutions:
            total_res = res**self.grid_dim
            table_size = min(total_res, self.max_entry)
            # +/- 10**-4 in InstantNGP paper
            table = torch.randn(int(table_size), self.feature_dim, device=self.device) * 0.0001
            self.hashtable.append(table)



    def forward(self, x):
        out_feature = []
        for lvl in range(self.num_lvl):
            coord = self.to_hash_space(x, self.resolutions[lvl])
            floor_corner = torch.floor(coord)
            corners = self.get_corner(floor_corner).to(torch.long)
            feature_index = self.hash(corners, self.hashtable[lvl].shape[0], self.resolutions[lvl])
            flat_feature_index = feature_index.to(torch.long).flatten()
            corner_feature = torch.reshape(self.hashtable[lvl][flat_feature_index],
                                           (corners.shape[0], corners.shape[1], self.feature_dim))
            weights = self.interpolation_weights(coord - floor_corner)
            # weights = self.alt_weights(corners, coord)
            weights = torch.stack([weights, weights], -1)
            weighted_feature = corner_feature * weights
            summed_feature = weighted_feature.sum(-2)
            out_feature.append(summed_feature)
        return torch.cat(out_feature, -1)

    def to_hash_space(self, x, resolution):
        # don't want the (res-1, res-1) corner. Easier for later get_corner()
        return torch.clip(x * (resolution - 1), 0, resolution - 1.0001)
    def interpolation_weights(self, diff):
        ones = torch.ones_like(diff, device=self.device)
        minus_x = (ones - diff)[..., 0]
        x = diff[..., 0]
        minus_y = (ones - diff)[..., 1]
        y = diff[..., 1]       
        if self.grid_dim == 2:
            pass 
            
        else: 
            minus_z = (ones - diff)[..., 2]
            z = diff[..., 2]
            stacks = torch.stack([
                minus_x * minus_y * minus_z, # 000
                minus_x * minus_y * z,       # 001 (只变 z)
                minus_x * y * minus_z,       # 010 (只变 y)
                minus_x * y * z,             # 011
                x * minus_y * minus_z,       # 100 (只变 x)
                x * minus_y * z,             # 101
                x * y * minus_z,             # 110
                x * y * z                    # 111
            ], -1)
            return stacks

    def alt_weights(self, corner, coord):
        # change diag accoring to dim
        diag_length = torch.full_like(coord[:, 0], 2.**(1/2), device=self.device)
        w = torch.empty(corner.shape[0], corner.shape[1], device=self.device)
        for c in range(corner.shape[1]):
            dist = torch.norm(corner[:, c, :] - coord, dim=1)
            w[:, c] = diag_length - dist
        normed_w = torch.nn.functional.normalize(w, p=1)
        return normed_w

    def hash(self, x, num_entry, res):
        if num_entry != self.max_entry:
            index = 0
            for i in range(self.grid_dim):
                index += x[..., i] * res**i
            return index
        else:
            _sum = 0
            for i in range(self.grid_dim):
                _sum = _sum ^ (x[..., i] * self.prime[i])
            index = _sum % num_entry
            return index

    def get_corner(self, floor_corner):
        num_entry = floor_corner.shape[0]
        if self.grid_dim == 2:
            # 2D 也建议按标准顺序: 00, 01, 10, 11
            c00 = floor_corner
            c01 = floor_corner + torch.tensor([0, 1], device=self.device) # y+1
            c10 = floor_corner + torch.tensor([1, 0], device=self.device) # x+1
            c11 = floor_corner + torch.tensor([1, 1], device=self.device)
            # 你的 2D weights 是怎么写的？如果不确定，建议这里也显式写出来
            return torch.stack([c00, c01, c10, c11], -2)
        else:
            # 3D 修正版：严格对应你的新 weights
            c000 = floor_corner
            c001 = floor_corner + torch.tensor([0, 0, 1], device=self.device)
            c010 = floor_corner + torch.tensor([0, 1, 0], device=self.device)
            c011 = floor_corner + torch.tensor([0, 1, 1], device=self.device)
            c100 = floor_corner + torch.tensor([1, 0, 0], device=self.device)
            c101 = floor_corner + torch.tensor([1, 0, 1], device=self.device)
            c110 = floor_corner + torch.tensor([1, 1, 0], device=self.device)
            c111 = floor_corner + torch.tensor([1, 1, 1], device=self.device)
            stacks = torch.stack([c000, c001, c010, c011, c100, c101, c110, c111], -2)
            return stacks

        
class Decoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 activation,
                 last_activation,
                 bias,
                 num_layers=6,
                 hidden_dim=128,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = self.fetch_activation(activation)
        self.last_activation = self.fetch_activation(last_activation)
        self.bias = bias
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            else:
                h = self.activation(l(h))
        out = self.out_layer(h)
        return out


    def fetch_activation(self, activation):
        if activation == "Relu":
            return nn.ReLU()
        elif activation == "Linear":
            return nn.Identity()
        elif activation == "Sigmoid":
            return nn.Sigmoid()
        elif activation == "Tanh":
            return nn.Tanh()
        else:
            print(f"Unknown activation {activation}")
            return nn.ReLU()
        
class NGPNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这里的 config 可以包含 NGP 的所有参数

        # 初始化哈希网格编码器
        self.grid = Grid(
            feature_dim=config.feature_dim,
            grid_dim=config.grid_dim,
            num_lvl=config.num_lvl,
            max_res=config.max_res,
            min_res=config.min_res,
            hashtable_power=config.hashtable_power,
            force_cpu=config.force_cpu
        )

        # 计算 Decoder 的输入维度：所有层级的特征维度之和
        decoder_input_dim = (config.num_lvl * config.feature_dim) +1
 
        # 初始化 Decoder (MLP)
        self.decoder = Decoder(
            input_dim=decoder_input_dim,
            output_dim=config.output_dim,
            activation=config.activation,
            last_activation=config.last_activation,
            bias=config.bias,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim
        )

    def forward(self, x):
        t = x[:, 0:1] 
        xyz = x[:, 1:] 
        xyz_normalized = (xyz + 1) / 2.0
        encoded_features = self.grid(xyz_normalized) 
        final_input_to_decoder = torch.cat([t, encoded_features], dim=-1)
        output = self.decoder(final_input_to_decoder)
        return output
    
    
    
import torch
import torch.nn as nn
import numpy as np

# 从 utils.py 提取的核心哈希函数
def hash_func(coords, log2_hashmap_size):
    '''
    coords: N x D coordinates
    '''
    # 这里的质数用于处理不同维度的哈希碰撞，支持高达 7 维的坐标
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result

# 从 utils.py 提取的体素顶点计算
def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x D
    bounding_box: min and max coordinates
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    # 确保坐标在包围盒内，虽然你的数据已经归一化，但加一层保险
    keep_mask = xyz == torch.max(torch.min(xyz, box_max), box_min)
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution
    
    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    
    voxel_indices = bottom_left_idx.unsqueeze(1) # B x 1 x D
    
    # 生成超立方体的顶点偏移 (支持任意维度)
    dim = xyz.shape[-1]
    # 生成类似于 [[0,0,0], [0,0,1], ...] 的二进制偏移
    import itertools
    offset_list = list(itertools.product([0, 1], repeat=dim))
    BOX_OFFSETS = torch.tensor(offset_list, device=xyz.device, dtype=torch.int) # 2^D x D
    
    voxel_indices = voxel_indices + BOX_OFFSETS # B x 2^D x D
    
    hashed_voxel_indices = hash_func(voxel_indices, log2_hashmap_size)
    
    return voxel_min_vertex, hashed_voxel_indices, keep_mask

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        # 确保 bounding_box 是 tensor
        if not isinstance(bounding_box[0], torch.Tensor):
            self.bounding_box = (torch.tensor(bounding_box[0]), torch.tensor(bounding_box[1]))
        else:
            self.bounding_box = bounding_box
            
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # 自定义均匀初始化，根据 NGP 论文建议
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def forward(self, x):
        # x: B x D (你的情况是 B x 4)
        # 确保 x 和 bounding_box 在同一个设备
        box_min = self.bounding_box[0].to(x.device)
        box_max = self.bounding_box[1].to(x.device)
        
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            
            # 获取体素顶点和哈希索引
            voxel_min_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices(
                x, (box_min, box_max), resolution, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices) # B x 2^D x Features

            # 进行多线性插值 (D-linear interpolation)
            # 为了简单适配任意维度（3D或4D），我们需要计算相对位置
            grid_size = (box_max - box_min) / resolution
            rel_coords = (x - voxel_min_vertex) / grid_size # B x D，在 [0,1] 之间
            
            # 手动实现多线性插值稍微复杂，这里针对 4D/3D 做一个通用的递归插值或直接用 grid_sample 的思路
            # 鉴于你的数据是 4D，下面的逻辑是一个通用的加权求和
            
            # Weights: B x 2^D
            n_dim = x.shape[-1]
            weights = torch.ones((x.shape[0], 2**n_dim), device=x.device)
            
            # 生成权重的逻辑：对于每个顶点的每个维度，如果该维度是1，乘 rel_coords，如果是0，乘 (1-rel_coords)
            # BOX_OFFSETS 顺序对应 weights 的列顺序
            # 重新生成 offset 列表以保持一致性
            import itertools
            offset_list = list(itertools.product([0, 1], repeat=n_dim))
            
            for idx, offset in enumerate(offset_list):
                for dim in range(n_dim):
                    if offset[dim] == 1:
                        weights[:, idx] *= rel_coords[:, dim]
                    else:
                        weights[:, idx] *= (1.0 - rel_coords[:, dim])
            
            # Weighted sum: (B x 2^D x 1) * (B x 2^D x F) -> sum over 2^D -> B x F
            x_embedded = (weights.unsqueeze(-1) * voxel_embedds).sum(dim=1)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)

class HashCoordNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, init_features=64, num_res=2):
        super(HashCoordNet, self).__init__()
        
        # 定义包围盒，你的数据被缩放到 [-1, 1]
        # 注意：包含时间维度 t，所以是 4 维
        bounding_box = (torch.tensor([-1.0]*in_channels), torch.tensor([1.0]*in_channels))
        
        # 1. 核心：Hash Encoding 层
        # base_resolution 和 finest_resolution 可以根据显存和精度需求调整
        self.embedder = HashEmbedder(bounding_box, n_levels=16, n_features_per_level=2, 
                                     log2_hashmap_size=18, base_resolution=8, finest_resolution=256)
        
        embedding_dim = self.embedder.out_dim # 16 * 2 = 32
        
        # 2. MLP 解码器 (根据 NGP，通常是小的 MLP)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels) # 输出单通道值
        )

    def forward(self, x):
        # x: [Batch, 4] (t, x, y, z)
        embed = self.embedder(x)
        out = self.net(embed)
        return out