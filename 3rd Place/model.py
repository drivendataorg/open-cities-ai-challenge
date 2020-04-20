'''
Created on Nov 22, 2019

@author: Michal.Busta at gmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from efficientnet_pytorch import EfficientNet


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True

class ContextBlock2d(nn.Module):

  def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=8):
    super(ContextBlock2d, self).__init__()
    assert pool in ['avg', 'att']
    assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
    assert len(fusions) > 0, 'at least one fusion should be used'
    self.inplanes = inplanes
    self.planes = planes
    self.pool = pool
    self.fusions = fusions
    if 'att' in pool:
      self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
      self.softmax = nn.Softmax(dim=2)
    else:
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    if 'channel_add' in fusions:
      self.channel_add_conv = nn.Sequential(
          nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
          nn.LayerNorm([self.planes // ratio, 1, 1]),
          nn.ReLU(inplace=True),
          nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
      )
    else:
      self.channel_add_conv = None
    if 'channel_mul' in fusions:
      self.channel_mul_conv = nn.Sequential(
          nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
          nn.LayerNorm([self.planes // ratio, 1, 1]),
          nn.ReLU(inplace=True),
          nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
      )
    else:
      self.channel_mul_conv = None
    self.reset_parameters()

  def reset_parameters(self):
    #if self.pool == 'att':
    #  nn.init.kaiming_init(self.conv_mask, mode='fan_in')
    #  self.conv_mask.inited = True

    if self.channel_add_conv is not None:
      last_zero_init(self.channel_add_conv)
    if self.channel_mul_conv is not None:
      last_zero_init(self.channel_mul_conv)

  def spatial_pool(self, x):
    batch, channel, height, width = x.size()
    if self.pool == 'att':
      input_x = x
      # [N, C, H * W]
      input_x = input_x.view(batch, channel, height * width)
      # [N, 1, C, H * W]
      input_x = input_x.unsqueeze(1)
      # [N, 1, H, W]
      context_mask = self.conv_mask(x)
      # [N, 1, H * W]
      context_mask = context_mask.view(batch, 1, height * width)
      # [N, 1, H * W]
      context_mask = self.softmax(context_mask)#softmax操作
      # [N, 1, H * W, 1]
      context_mask = context_mask.unsqueeze(3)
      # [N, 1, C, 1]
      context = torch.matmul(input_x, context_mask)
      # [N, C, 1, 1]
      context = context.view(batch, channel, 1, 1)
    else:
      # [N, C, 1, 1]
      context = self.avg_pool(x)

    return context

  def forward(self, x):
      # [N, C, 1, 1]
    context = self.spatial_pool(x)

    if self.channel_mul_conv is not None:
      # [N, C, 1, 1]
      channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
      out = x * channel_mul_term
    else:
      out = x
    if self.channel_add_conv is not None:
      # [N, C, 1, 1]
      channel_add_term = self.channel_add_conv(context)
      out = out + channel_add_term

    return out


class ConvGnUp2d(nn.Module):
  
  def __init__(self, in_channel, out_channel, num_group=32, kernel_size=3, padding=1, stride=1):
    super(ConvGnUp2d, self).__init__()
    self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
    self.gn   = nn.GroupNorm(num_group,out_channel)

  def forward(self,x):
    x = self.conv(x)
    x = self.gn(x)
    x = F.relu(x, inplace=True)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
    return x
    
class SELayer(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SELayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction, bias=False),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)
    
class ASPPConv(nn.Module):
  def __init__(self, in_channel, out_channel, dilation):
    super(ASPPConv, self).__init__()
    self.module = nn.Sequential(
      nn.Conv2d(in_channel, out_channel, 3, padding=dilation, dilation=dilation, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True)
    )
  def forward(self,x):
    x = self.module(x)
    return x

class ASPPPool(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(ASPPPool, self).__init__()
    self.module = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    batch_size,C,H,W = x.shape
    x = self.module(x)
    x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True)
    return x

class ASPP(nn.Module):
  def __init__(self, in_channel, out_channel=256,rate=[6,12,18], dropout_rate=0):
    super(ASPP, self).__init__()

    self.atrous0 = nn.Sequential(
      nn.Conv2d(in_channel, out_channel, 1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True)
    )
    self.atrous1 = ASPPConv(in_channel, out_channel, rate[0])
    self.atrous2 = ASPPConv(in_channel, out_channel, rate[1])
    self.atrous3 = ASPPConv(in_channel, out_channel, rate[2])
    self.atrous4 = ASPPPool(in_channel, out_channel)

    self.combine = nn.Sequential(
      nn.Conv2d(5 * out_channel, out_channel, 1, bias=False),
      nn.BatchNorm2d(out_channel),
      nn.ReLU(inplace=True),
      nn.Dropout(dropout_rate)
    )

  def forward(self, x):

    x = torch.cat([
      self.atrous0(x),
      self.atrous1(x),
      self.atrous2(x),
      self.atrous3(x),
      self.atrous4(x),
    ],1)
    x = self.combine(x)
    return x
    
class Mish(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
    return x *( torch.tanh(F.softplus(x)))
  
def upsize_add(x, lateral):
  return F.interpolate(x, size=lateral.shape[2:], mode='nearest') + lateral

class EfficientFPN(nn.Module):
    
  def __init__(self, encoder_name='efficientnet-b5', use_pretrained=True, 
               output_scale=1, use_attention = True, use_ASPP = True, use_context_block=False, use_mish=False, drop_connect_rate = 0.2):
    
    super(EfficientFPN, self).__init__()
    if use_pretrained:
      e = EfficientNet.from_pretrained(encoder_name)
    else:
      e = EfficientNet.from_name(encoder_name)
      
    self.encoder_name = encoder_name
    self.output_scale = output_scale
    
    self._bn0 = e._bn0
    self._conv_stem = e._conv_stem
    self._global_params = e._global_params
    self.drop_connect_rate = drop_connect_rate 
    
    self.mish = Mish()      
           
    self._blocks = e._blocks
    self._conv_head = e._conv_head
    self._bn1 = e._bn1
    e = None  #dropped
    
    self.lat_filters = 1280
    self.lat_filters1 =  112
    self.lat_filters2 =  40
    self.lat_filters3 =  24
    if encoder_name == 'efficientnet-b1':
      self.lat_filters = 1280
      self.lat_filters1 =  112
      self.lat_filters2 =  40
      self.lat_filters3 =  24
    if encoder_name == 'efficientnet-b2':
      self.lat_filters = 1408
      self.lat_filters1 =  120
      self.lat_filters2 =  48
      self.lat_filters3 =  24
    if encoder_name == 'efficientnet-b3':
      self.lat_filters = 1536
      self.lat_filters1 =  136
      self.lat_filters2 =  48
      self.lat_filters3 =  32
        
    #---
    self.lateral0 = nn.Conv2d( self.lat_filters,  64,  kernel_size=1, padding=0, stride=1)
    self.lateral1 = nn.Conv2d( self.lat_filters1, 64,  kernel_size=1, padding=0, stride=1)
    self.lateral2 = nn.Conv2d( self.lat_filters2, 64,  kernel_size=1, padding=0, stride=1)
    self.lateral3 = nn.Conv2d( self.lat_filters3, 64,  kernel_size=1, padding=0, stride=1)
    
    
    self.top1 = nn.Sequential(
      ConvGnUp2d(64, 64),
      ConvGnUp2d(64, 64),
    )
    self.top2 = nn.Sequential(
      ConvGnUp2d(64, 64),
    )
    
    self.top3 = nn.Sequential(
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    )
    self.use_context_block = use_context_block
    if use_context_block:
      if self.encoder_name == 'efficientnet-b2':
        self.ctx = ContextBlock2d(64*3, 64)
        self.top4 = nn.Sequential(
          nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(64), #TODO BatchNorm2d(64), 
          Mish(),
        )
      else:
        self.ctx = ContextBlock2d(64*3, 32)
        self.top4 = nn.Sequential(
          ContextBlock2d(64*3, 64),
          nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(64), #TODO BatchNorm2d(64), 
          Mish(),
        )
    elif use_mish:
      print('mish')
      self.top4 = nn.Sequential(
        nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64), #TODO BatchNorm2d(64), 
        Mish()
      )
    else:
      self.top4 = nn.Sequential(
        nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64), #TODO BatchNorm2d(64), 
        nn.ReLU(inplace=True),
      )
    
    if use_attention:
      self.conv_att = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        
    self.use_attention = use_attention
    
    if use_ASPP:
      self.aspp = ASPP(192, 192, rate=[4,8,12], dropout_rate=0.1)
    
    self.bce_logit = nn.Conv2d(64, 2,kernel_size=1)
    self.dice = nn.Conv2d(64, 1,kernel_size=1, bias=False)
    self.use_ASPP = use_ASPP
    self.dropout = nn.Dropout(p=0.2)
    self._avg_pooling = nn.AdaptiveAvgPool2d(1)
    self._fc = nn.Linear(self._bn1.num_features, 2)
    self._fc2 = nn.Linear(self._bn1.num_features, 1)

  def forward(self, x, debug=False):
    
    bs = x.size(0)
    if self.encoder_name == 'efficientnet-b2':
      with torch.no_grad():
        x = self.mish(self._bn0(self._conv_stem(x)))
    else:
      x = self.mish(self._bn0(self._conv_stem(x)))
    for idx, block in enumerate(self._blocks):
      drop_connect_rate = 0.2
      if idx > 3:
        drop_connect_rate = self.drop_connect_rate
      if drop_connect_rate:
        drop_connect_rate *= float(idx) / len(self._blocks)
      x = block(x, drop_connect_rate=drop_connect_rate)
      if x.shape[1] == self.lat_filters3:
        x1=x
      elif x.shape[1] == self.lat_filters2:
        x2=x
      elif x.shape[1] == self.lat_filters1:
        x3=x
         
    #x = relu_fn(self.last(x));x4=x #; print('last  ',x.shape)
    x = self.mish(self._bn1(self._conv_head(x)));x4=x
    
    
    xc = self._avg_pooling(x)
    xc = xc.view(bs, -1)
    xc = self.dropout(xc)
    xs = self._fc2(xc) 
    xc = self._fc(xc)
    
    # segment
    t0 = self.lateral0(x4)
    t1 = upsize_add(t0, self.lateral1(x3))
    #t1 = upsize_conv_add(x4, self.upconv1, self.lateral1(x3)) #16x16
    t2 = upsize_add(t1, self.lateral2(x2)) #32x32
    #t2 = upsize_conv_add(t1, self.upconv2, self.lateral2(x2))
    t3 = upsize_add(t2, self.lateral3(x1)) #64x64
    #t3 = upsize_conv_add(t2, self.upconv3, self.lateral3(x1)) #64x64

    t1 = self.top1(t1) #128x128
    t2 = self.top2(t2) #128x128
    t3 = self.top3(t3) #128x128

    t = torch.cat([t1,t2,t3],1)
    t = self.dropout(t)
    t0 = t
    
    if self.use_context_block:
      t0 = self.ctx(t0)
      
    if self.use_ASPP:
      t0 = self.aspp(t0)
      t = self.top4(t0)
    else:
      t = self.top4(t)
      t0 = t
    
    if self.use_attention: 
      x_atts = torch.tanh(self.conv_att(t))
      t = t * x_atts
        
    bceout = self.bce_logit(t)
    dice = self.dice(t)
    
    if self.output_scale != 1:
      bceout = F.interpolate(bceout, scale_factor=self.output_scale, mode='bilinear', align_corners=False)
      dice = F.interpolate(dice, scale_factor=self.output_scale, mode='bilinear', align_corners=False)
    
    return bceout, dice, xc, xs
  
  def freeze_bn(self):
    '''Freeze BatchNorm layers.'''
    self._bn0.training = False 
    
def load_net(sp, net):
  sp = sp['state_dict']
  for k, v in net.state_dict().items():
    try:
      param = sp[k]
      v.copy_(param)
      #print('{0} - {1}'.format(v.max(), k))
    except:
      if k == 'stem.0.weight':
        try:
          param = param.mean(1)
          param = param.unsqueeze(1)
          v.copy_(param)
        except:
          import traceback
          traceback.print_exc()
                