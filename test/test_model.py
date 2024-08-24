import mindspore.nn as nn
import mindspore.ops.operations as ops
import mindspore

class SingleInputNet(nn.Cell):
    def __init__(self):
        super(SingleInputNet, self).__init__()
        # mindspore卷积默认不加bias
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, pad_mode='valid', has_bias=True)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, pad_mode='valid', has_bias=True)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Dense(320, 50)
        self.fc2 = nn.Dense(50, 10)

    def construct(self, x):
        x = ops.ReLU()(nn.MaxPool2d(2, 2)(self.conv1(x)))
        x = ops.ReLU()(nn.MaxPool2d(2, 2)(self.conv2_drop(self.conv2(x))))
        x = x.reshape((-1, 320))
        x = ops.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return ops.LogSoftmax(axis=1)(x)

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

class MultipleInputNetDifferentDtypes(nn.Cell):
    def __init__(self):
        super(MultipleInputNetDifferentDtypes, self).__init__()
        self.fc1a = nn.Dense(300, 50)
        self.fc1b = nn.Dense(50, 10)

        self.fc2a = nn.Dense(300, 50)
        self.fc2b = nn.Dense(50, 10)

    def construct(self, x1, x2):
        x1 = ops.ReLU()(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = ops.ReLU()(self.fc2a(x2.astype(mindspore.float32)))
        x2 = self.fc2b(x2)
        x = ops.Concat(0)((x1, x2))
        return ops.LogSoftmax(axis=1)(x)