from mindspore import nn, Tensor, float32, context
from collections import OrderedDict
import numpy as np

def summary(net, input_size, batch_size=-1, device_target="GPU", device_id=0, dtypes=None):
    """
    Print a summary of the network.
    """
    result, params_info = summary_string(
        net, input_size, batch_size, device_target, device_id, dtypes)
    print(result)
    return params_info

def summary_string(net, input_size, batch_size=-1, device_target="GPU", device_id=0, dtypes=None):
    """
    Print a summary of the network.
    """
    context.set_context(device_target=device_target, device_id=device_id)

    # create properties
    summary = OrderedDict()
    hooks = []
    summary_str = ''
    def register_hook(cell):
        def hook(cell, inputs, outputs):
            class_name = cell.__class__.__name__
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx + 1}"

            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(inputs[0].shape)
            summary[m_key]["input_shape"][0] = batch_size

            if isinstance(outputs, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.shape)[1:] for o in outputs]
            else:
                summary[m_key]["output_shape"] = list(outputs.shape)
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(cell, "weight") and hasattr(cell.weight, "size"):
                params += cell.weight.numel()
                summary[m_key]["trainable"] = cell.weight.requires_grad
            if hasattr(cell, "bias") and hasattr(cell.bias, "size"):
                params += cell.bias.numel()
            summary[m_key]["nb_params"] = params

        if not isinstance(cell, (nn.SequentialCell, nn.CellList)):
            hooks.append(cell.register_forward_hook(hook))

    # register hook
    net.apply(register_hook)

    if dtypes is None:
        dtypes = [float32] * len(input_size)

    # input_size may have multiple inputs
    if isinstance(input_size, tuple):
        input_size = [input_size]

    x = [Tensor(np.random.rand(2, *in_size), dtype=dtype) 
         for in_size, dtype in zip(input_size, dtypes)]
    
    # make a forward pass
    net(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # formated summary
    summary_str = ""
    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    
    for layer in summary:
        # output_shape, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"
    
    # assume 4 bytes/number (float on GPU).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"

    # return summary
    return summary_str, (total_params, trainable_params)