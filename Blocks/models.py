import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import torch.nn.functional as F

class ActNorm(nn.Module):
  def __init__(self, num_channels, height, width):
    super(ActNorm, self).__init__()
    #learn logscale for stability
    # Per channel scale and bias parameter
    self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True )
    self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1), requires_grad=True)
    self.initialised = False
    self.h = height
    self.w = width

  def forward(self, x):
    # Initialised so that first minibatch has mean 0 and var 1
    if not self.initialised:
      self.bias.data = -torch.mean(x, (0,2,3), keepdim=True)
      self.log_scale.data = -torch.log(torch.std(x, (0,2,3), keepdim=True))
      self.initialised=True

    return (x * torch.exp(self.log_scale)) + self.bias, self.h * self.w * self.log_scale.sum()

  def inverse(self, z):
    return (z  - self.bias) * torch.exp(-self.log_scale)



class Invertible1x1Conv2d(nn.Module):
  def __init__(self, n_channels):
    super(Invertible1x1Conv2d, self).__init__()
    # Random rotation matrix
    W = torch.linalg.qr(torch.rand(n_channels, n_channels, dtype=torch.float32))[0]
    P, L, U = torch.linalg.lu(W)
    s = U.diag()
    self.P = nn.Parameter(P, requires_grad=False)
    self.L = nn.Parameter(L, requires_grad=True)
    self.U = nn.Parameter(U-s.diag(), requires_grad=True)
    self.s = nn.Parameter(s, requires_grad=True)

    self.L_mask = torch.tril(torch.ones(self.L.shape, dtype=torch.bool), diagonal=-1)
    self.U_mask = torch.triu(torch.ones(self.U.shape, dtype=torch.bool), diagonal=1)
    self.identity = torch.eye(self.L.shape[0], dtype=torch.float32)


    self.weights_updated = True
    self.W_inv = None

  def forward(self, x):
    self.identity = self.identity.to(x.device)
    self.L_mask = self.L_mask.to(x.device)
    self.U_mask = self.U_mask.to(x.device)


    W = torch.matmul(self.P, torch.matmul(self.L*self.L_mask +self.identity, (self.U*self.U_mask + self.s.diag() ))).unsqueeze(2).unsqueeze(3)

    log_det_W = torch.log(self.s.abs()).sum() # Migth need to learn log_scale instead?

    return F.conv2d(x, W), x.shape[2]* x.shape[3] * log_det_W

  def inverse(self, z):

    if self.weights_updated is True:
      self.W_inv = torch.matmul(self.P, torch.matmul(self.L*self.L_mask +self.identity, (self.U*self.U_mask + self.s.diag() ))).inverse().unsqueeze(2).unsqueeze(3)
      self.weights_updated = False


    return F.conv2d(z, self.W_inv)

## TODO convert self.s to self.log_s for stability


class WeightNormConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)

class ResNetBlock(nn.Module):
  def __init__(self, hidden_features):
    super(ResNetBlock, self).__init__()
    self.block = nn.Sequential(
        WeightNormConv2d(hidden_features, hidden_features, kernel_size=1, stride=1, padding=0 ),
        nn.ReLU(),
        WeightNormConv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1 ),
        nn.ReLU(),
        WeightNormConv2d(hidden_features, hidden_features, kernel_size=1, stride=1, padding=0 )
    )

  def forward(self, x):
    return x + self.block(x)

class ResNet(nn.Module):
  def __init__(self, in_channels=3, hidden_features=512, num_blocks=1):
    super(ResNet, self).__init__()

    layers = [WeightNormConv2d(in_channels, hidden_features, kernel_size=3, stride=1, padding=1), nn.ReLU()]
    for _ in range(num_blocks-2):
      layers.append(ResNetBlock( hidden_features))
    layers.append(nn.ReLU())
    layers.append(WeightNormConv2d(hidden_features, 2*in_channels, kernel_size=3, stride=1, padding=1)) # Double the in_channels out for s and t
    self.net = nn.Sequential(*layers)

    self.net[0].conv.weight.data.normal_(0, 0.05)
    self.net[0].conv.bias.data.zero_()

    self.net[-1].conv.weight.data.normal_(0, 0.05)
    self.net[-1].conv.bias.data.zero_()


  def forward(self, x):
    return self.net(x)

class ConvBlock(nn.Module):
  def __init__(self, in_channels, hidden_features=512):
    super(ConvBlock, self).__init__()
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

class AffineCoupling(nn.Module):
    def __init__(self, n_channels):
        super(AffineCoupling, self).__init__()
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

      return y, log_scale.sum().to(device=x.device)

    def inverse(self, y):
      y_a, y_b = y.chunk(2, dim=1)

      log_scale, shift = self.net(y_b).chunk(2, dim=1)
      log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale

      x_a = (y_a - shift) * torch.exp(-log_scale)
      x_b = y_b

      x = torch.cat([x_a, x_b], dim=1)

      return x


class GlowStep(nn.Module):
  def __init__(self, n_channels, h, w):
    super(GlowStep, self).__init__()
    self.layers = nn.ModuleList([
        ActNorm(n_channels, 3, 3),
        Invertible1x1Conv2d(n_channels),
        AffineCoupling(n_channels),
    ])

  def forward(self, x):
    log_det_jacobian_total = torch.tensor(0.0).to(device=x.device)
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



class Glow(nn.Module):
  def __init__(self, n_flow_steps=3, n_flow_layers=3, n_channels=3, height=32, width=32):
    super(Glow, self).__init__()

    # self.preprocess = Preprocess()
    self.flow_layers = nn.ModuleList()


    for layer in range(1, n_flow_layers+1):
      flow_steps = nn.ModuleList()
      for _ in range(n_flow_steps):
        # print("channels:", n_channels*(4**(layer)))
        # print(height//(2*(layer)))
        # print(width//(2*(layer)))
        flow_steps.append(GlowStep(n_channels=n_channels*(4**(layer)),
                                        h=height//(2*(layer)),
                                        w=width//(2*(layer)))
        )
      self.flow_layers.append(flow_steps)



  def squeeze2d(self, x, factor=2):
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

  def unsqueeze2d(self, x, factor=2):
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


  def forward(self, x):

    log_det_jacobian_total = torch.tensor(0.0, device=x.device)
    z_i = x


    for layer in self.flow_layers:
      z_i = self.squeeze2d(z_i)
      for flow_step in layer:
        z_i, log_det_jacobian = flow_step(z_i)
        log_det_jacobian_total += log_det_jacobian


    z_0 = z_i
    for n_layers in range(len(self.flow_layers)):
      z_0 =  self.unsqueeze2d(z_0)

    # TODO come back and store the values of z_i in a matrix

    return z_0, log_det_jacobian_total



  def inverse(self, z_0):
    z_i = z_0

    for n_layers in range(len(self.flow_layers)):
      z_i =  self.squeeze2d(z_i)


    for layer in self.flow_layers[::-1]:
      for flow_step in layer[::-1]:
        z_i = flow_step.inverse(z_i)
      z_i = self.unsqueeze2d(z_i)

    x = z_i


    return x
