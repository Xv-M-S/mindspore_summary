import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common.initializer import Normal
from netsummary import summary, summary_string
# from mindspore.info_summary import summary

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

print(model.trainable_params())
trainable_param = 0
for param in model.get_parameters():
    trainable_param += param.data.numel() 
    print(param.name, param.data.shape)
print(trainable_param)

print(model.untrainable_params())

total_params = 0
for param in model.get_parameters():
    total_params += param.data.numel() 
    print(param.name, param.data.shape)
print(total_params)

summary(model, (1, 28, 28),device_target="CPU")