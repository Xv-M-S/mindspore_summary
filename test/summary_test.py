import unittest
import mindspore as ms
from mindspore.common.initializer import Normal
import mindspore.nn as nn
import mindspore.ops as ops
from netsummary import summary, summary_string
# from mindspore.info_summary import net_summary, summary_string
from test_model import SingleInputNet, MultipleInputNet, MultipleInputNetDifferentDtypes


class MindSporeModelSummaryTests(unittest.TestCase):
    def test_single_input(self):
        net = SingleInputNet()
        input_shape = (1, 28, 28)
        total_params, trainable_params = summary(net, input_shape, device_target="CPU")
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)
    
    def test_multiple_input(self):
        model = MultipleInputNet()
        input1 = (1, 300)
        input2 = (1, 300)
        total_params, trainable_params = summary(
            model, [input1, input2], device_target="CPU")
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

    def test_single_layer_network(self):
        model = nn.Dense(2, 5)
        input = (1, 2)
        total_params, trainable_params = summary(model, input, device_target="CPU")
        self.assertEqual(total_params, 15)
        self.assertEqual(trainable_params, 15)

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = (1, 300)
        input2 = (1, 300)
        dtypes = [ms.float32, ms.int64]
        total_params, trainable_params = summary(
            model, [input1, input2], device_target="CPU", dtypes=dtypes)
        self.assertEqual(total_params, 31120)
        self.assertEqual(trainable_params, 31120)

class torchsummarystringTests(unittest.TestCase):
    def test_single_input(self):
        model = SingleInputNet()
        input = (1, 28, 28)
        result, (total_params, trainable_params) = summary_string(
            model, input, device_target="CPU")
        self.assertEqual(type(result), str)
        self.assertEqual(total_params, 21840)
        self.assertEqual(trainable_params, 21840)

if __name__ == '__main__':
    unittest.main()