import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18 as r18


class simple_net(nn.Module):
    def __init__(self, img_side, num_conv_layers=2, num_fc_layers=2, kernel_sides=[5,5], num_conv_features=[16,16], num_fc_features=[128], bias=True):
        super(simple_net, self).__init__()

        self.img_side = img_side

        num_conv_features = [1] + num_conv_features
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(num_conv_features[i], num_conv_features[i+1], kernel_sides[i], padding=0, bias=bias) for i in range(num_conv_layers)])

        out_side = img_side - np.array(kernel_sides).sum() + num_conv_layers
        num_fc_features = [out_side**2 * num_conv_features[-1]] +  num_fc_features + [2]
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(num_fc_features[i], num_fc_features[i+1]) for i in range(num_fc_layers)])
 
    def forward(self, x):
        batch_size = x.shape[0]
        for layer in self.conv_layers:
            x = layer(x)
            x = torch.relu(x)
        x = x.reshape(batch_size, -1)
        for l, layer in enumerate(self.fc_layers):
            x = layer(x)
            x = torch.relu(x) if l < len(self.fc_layers) - 1 else torch.sigmoid(x)
        return x

    def train(self, generator, loss, optimizer, batch_size=32, lr=1e-3, device='cpu', steps=100):
        for i in range(steps):
            b, l, _ = generator.sample_batch(batch_size)
            out = self.forward(b.to(device).float())
            L = loss(out, l.to(device).float()).mean()
            L.backward()
            optimizer.step()
            optimizer.zero_grad()


class resnet18(r18):
    def __init__(self):
        super(resnet18, self).__init__()



if __name__=='__main__':
    img_side=128
    x = torch.rand(32,1,img_side, img_side)
    my_net = simple_net(img_side)
    out = my_net.forward(x)
