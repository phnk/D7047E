import torch

# from: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/filter_visualizer.ipynb
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True)
    def close(self):
        self.hook.remove()
