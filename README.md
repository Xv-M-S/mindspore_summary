# mindspore_summary
Torch has a neat API to view the visualization of the model which is very helpful while debugging your network. Here is a code to try and mimic the same in MindSpore. The aim is to provide information complementary to, what is not provided by print(your_model) in MindSpore.

# Usage
- Clone the repository: 'git clone https://github.com/Xv-M-S/mindspore_summary.git'
- Install the package using 'python setup.py install'

```python
from netsummary import summary,summary_string
summary(your_model, input_size=(channels, H, W))
```
- Note that the input_size is required to make a forward pass through the network.

# Examples
## SingleInputNet
```CNN for MNIST
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common.initializer import Normal
from netsummary import summary, summary_string

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, pad_mode='valid', weight_init=Normal(0.0, 0.1), has_bias=True)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, pad_mode='valid', weight_init=Normal(0.0, 0.1), has_bias=True)
        self.conv2_drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Dense(320, 50, weight_init=Normal(0.0, 0.1))
        self.fc2 = nn.Dense(50, 10, weight_init=Normal(0.0, 0.1))
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.log_softmax = nn.LogSoftmax(axis=1)

    def construct(self, x):
        x = self.relu(self.max_pool2d(self.conv1(x)))
        x = self.relu(self.max_pool2d(self.conv2_drop(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


model = Net()
summary(model, (1, 28, 28),device_target="CPU")
```

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 24, 24]             260
         MaxPool2d-2           [-1, 10, 12, 12]               0
              ReLU-3           [-1, 10, 12, 12]               0
            Conv2d-4             [-1, 20, 8, 8]           5,020
           Dropout-5             [-1, 20, 8, 8]               0
         MaxPool2d-6             [-1, 20, 4, 4]               0
              ReLU-7             [-1, 20, 4, 4]               0
           Flatten-8                  [-1, 320]               0
             Dense-9                   [-1, 50]          16,050
             ReLU-10                   [-1, 50]               0
            Dense-11                   [-1, 10]             510
       LogSoftmax-12                   [-1, 10]               0
              Net-13                   [-1, 10]               0
================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.09
Params size (MB): 0.08
Estimated Total Size (MB): 0.18
----------------------------------------------------------------
```

## MultipleInputNet

```python
class MultipleInputNet(nn.Cell):
    def __init__(self):
        super(MultipleInputNet, self).__init__()
        self.fc1a = nn.Dense(300, 50)
        self.fc1b = nn.Dense(50, 10)

        self.fc2a = nn.Dense(300, 50)
        self.fc2b = nn.Dense(50, 10)

    def construct(self, x1, x2):
        x1 = ops.ReLU()(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = ops.ReLU()(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = ops.Concat(0)((x1, x2))
        return ops.LogSoftmax(axis=1)(x)
```
```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
             Dense-1                [-1, 1, 50]          15,050
             Dense-2                [-1, 1, 10]             510
             Dense-3                [-1, 1, 50]          15,050
             Dense-4                [-1, 1, 10]             510
  MultipleInputNet-5                [-1, 1, 10]               0
================================================================
Total params: 31,120
Trainable params: 31,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.34
Forward/backward pass size (MB): 0.00
Params size (MB): 0.12
Estimated Total Size (MB): 0.46
----------------------------------------------------------------
```
# References
- Thanks to @sksq96

# License
- 'netsummary'is licensed under the Apache License 2.0.