import torch
from torch import nn
from torch.nn import functional as F

def squeeze2d( x, factor=2):
  '''
  Changes the shape of x from (Batch_size, Channels, Height, Width )
  to (Batch_size, 4*Channels, Height/2, Width/2).

  x: (Tensor) input
  factor (int): how many slices to split the data
  '''

  B, C, H, W = x.shape
  assert H % factor == 0 and W % factor == 0, 'Height and Width must be divisible by factor.'

  x = x.view(B, C, H // factor, factor, W // factor, factor)
  x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
  x = x.view(B, C * (factor**2), H // factor, W // factor)
  return x

def unsqueeze2d( x, factor=2):
  '''
  Reverses the Squeeze operation above.

  x: (Tensor) input
  factor (int): how many slices to split the data
  '''
  B, C, H, W = x.shape

  x = x.view(B, C // 4, factor, factor, H, W)
  x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
  x = x.view(B, C // (factor ** 2), H *factor, W * factor)
  return x

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
  def __init__(self, in_channels, hidden_features=512):
    super().__init__()
    layers =  [
      WeightNormConv2d(in_channels, hidden_features, kernel_size=3, padding=1),
      nn.ReLU(),
      WeightNormConv2d(hidden_features, hidden_features, kernel_size=1),
      nn.ReLU(),
      WeightNormConv2d(hidden_features, 2*in_channels, kernel_size=3, padding=1),
    ]
    self.net = nn.Sequential(*layers)

    self.net[0].conv.weight.data.normal_(0, 0.05)
    self.net[0].conv.bias.data.zero_()

    self.net[-1].conv.weight.data.normal_(0, 0.05)
    self.net[-1].conv.bias.data.zero_()

  def forward(self, x):
    return self.net(x)

class Preprocess(nn.Module):
    def __init__(self, bits=8, dequantize=True):
        super().__init__()
        self.bins = 2.**bits
        self.dequantize = dequantize
        
    def forward(self, x, train=True):
        if (train and self.dequantize) is False:
            return x
        y =  (x*255. + torch.rand_like(x))/self.bins
        return y
    
    
class Actnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = None
        self.bias = None
        self.h = None
        self.w = None
        self.register_buffer("initialised", torch.tensor(0, dtype=bool))
        
    def forward(self, x):
        if self.initialised == False:
            self.bias = nn.Parameter(-x.mean(dim=(0,2,3), keepdim=True), requires_grad=True)
            self.log_scale = nn.Parameter(-torch.log(x.std(dim=(0,2,3), keepdim=True)), requires_grad=True)
            
            _, _, self.w, self.h = x.shape
            
            self.initialised = ~self.initialised
            
        z = (x  + self.bias) * self.log_scale.exp()
        log_det_jacobian = self.h * self.w * self.log_scale.exp().abs().sum().unsqueeze(0)
        return z, log_det_jacobian
    
    def inverse(self, z):
        x = (z  * torch.exp(-self.log_scale)) - self.bias
        return x

class Inv1x1Conv(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
    
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        w_p, w_l, w_u = torch.linalg.lu(q)
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T
        


        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight
    
    def forward(self, x):
        _, _, height, width = x.shape
              
        
        weight = self.calc_weight().unsqueeze(2).unsqueeze(3)

        z = F.conv2d(x, weight)
        logdet = height * width * torch.sum(self.w_s)

        return z, logdet.unsqueeze(0)

    def inverse(self, z):
        weight_inv = self.calc_weight().inverse()
        
        return F.conv2d(z, weight_inv.unsqueeze(2).unsqueeze(3))
        
    
class AffineCoupling(nn.Module):
    def __init__(self, n_channels):
      super().__init__()
      self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
      self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
      # self.net = ResNet(in_channels=n_channels//2)
      self.net = ConvBlock(in_channels= n_channels//2)

    def forward(self, x):
      # Split the input in 2 channelwise
      x_a, x_b = x.chunk(2, dim=1)

      log_scale, shift = self.net(x_b).chunk(2, dim=1)
      log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale

      y_a = x_a * log_scale.exp() + shift
      y_b = x_b

      y = torch.cat([y_a, y_b], dim=1)

      return y, log_scale.view(x.shape[0], -1).sum(-1)

    def inverse(self, y):
      y_a, y_b = y.chunk(2, dim=1)

      log_scale, shift = self.net(y_b).chunk(2, dim=1)
      log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale

      x_a = (y_a - shift) * torch.exp(-log_scale)
      x_b = y_b

      x = torch.cat([x_a, x_b], dim=1)

      return x
    
class FlowStep(nn.Module):
    def __init__(self, n_channels ):
        super().__init__()
        self.layers = nn.ModuleList([
            Actnorm(),
            Inv1x1Conv(n_channels),
            AffineCoupling(n_channels),
        ])
        
    def forward(self, x):
        log_det_jacobian_total = torch.zeros(x.shape[0], device=x.device)
        z = x

        for layer in self.layers:
            z, log_det_jacobian = layer(z)
            log_det_jacobian_total += log_det_jacobian
        return z, log_det_jacobian_total.to(device=x.device)
    
    def inverse(self, z):
        x = z
        for layer in self.layers[::-1]:
            x = layer.inverse(x)
        return x
        
    
class FlowBlock(nn.Module):
    def __init__(self, n_channels, n_steps, split=True) -> None:
        super().__init__()
        self.flow_steps = nn.ModuleList()
        
        self.split = split
        
        for _ in range(n_steps):
            self.flow_steps.append(FlowStep(4*n_channels))
        
    def forward(self, x):
        # Squeeze
        z = squeeze2d(x)
        log_det_jacobian_total = torch.zeros(x.shape[0], device=x.device)
        
        # Flow
        for flow in self.flow_steps:
            z, log_det_jacobian = flow(z)
            log_det_jacobian_total += log_det_jacobian
        
        # Split
        if self.split is True:
            z_i, h_i = z.chunk(2, dim=1)
            return z_i, h_i, log_det_jacobian_total
        
        return z, 0 ,  log_det_jacobian_total
    
    def inverse(self, z, z_prev=None):
        
        # Invert split
        if z_prev is not None:
            z = torch.cat([z_prev, z], dim=1)

        
        # Invert flows
        for flow in self.flow_steps[::-1]:
            z = flow.inverse(z)
        
        # invert squeeze
        x = unsqueeze2d(z)
        return x
        
       
        
        
    
class Glow(nn.Module):
    def __init__(self, n_channels=3, n_steps=3, n_flow_blocks=3):
        super(Glow, self).__init__()

        self.preprocess = Preprocess()
        self.num_blocks = n_flow_blocks
        self.flow_layers = nn.ModuleList()
        
        for layer in range(0, n_flow_blocks-1):
            self.flow_layers.append(FlowBlock(n_channels=n_channels*2**(layer),
                                              n_steps=n_steps,
                                              split=True))
        self.flow_layers.append(FlowBlock(n_channels=n_channels*2**(n_flow_blocks-1),
                                              n_steps=n_steps,
                                              split=False))
        
    def forward(self, x, train=True):
        h = x
        z_list = []
        
        if train is True:
            h = self.preprocess(h)
            
        log_det_jacobian_total = torch.zeros(x.shape[0], device=x.device)
        for block in self.flow_layers:
            z, h, log_det_jacobian = block(h)
            z_list.append(z)
            log_det_jacobian_total += log_det_jacobian
        
        
        return z_list, log_det_jacobian_total
    
    def inverse(self, z_list):
        x = z_list.pop()
        
        for block in self.flow_layers[::-1]:
            z = z_list.pop() if block.split is True else None
            x = block.inverse(x,z)

        return x
    
    def list_to_z(self, z_list):
        z_0 = unsqueeze2d(z_list.pop())
        

        while z_list:
            z_0 = torch.cat((z_list.pop(), z_0), dim=1) # 6
            z_0 =  unsqueeze2d(z_0) # 3

        return z_0
    
    def z_to_list(self, input):
        z_list = []
        h = input

        for _ in range(len(self.flow_layers)-1):
            h =  squeeze2d(h) # 3
            z, h = torch.chunk(h,2, dim=1) # 6
            z_list.append(z)

        z_list.append(squeeze2d(h))
        return z_list
