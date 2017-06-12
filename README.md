# cnnvis-pytorch
visualization of CNN in PyTorch

this project is inspired by a summary of visualization methods in
[Lasagne examples](https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
), as well as [deep visualization toolbox](https://github.com/yosinski/deep-visualization-toolbox).

Visualization of CNN units in higher layers is important for my work, and currently (May 2017), I'm not
aware of any library with similar capabilities as the two mentioned above written for PyTorch.

Indeed I have some experience with deep visualization toolbox, which only supports Caffe.
However, it has very poor support for networks whose input size is not around 256x256x3
(standard size for ImageNet dataset, before cropping),
and indeed I need to visualize networks not having input of such size, such as networks trained on CIFAR-10, etc.
In addition, it can't support visualization techniques other than "deconvolution". Therefore, eventually,
converting PyTorch models to Caffe and then hacking the code of deep visualization toolbox to make it work
is probably not worthwhile.

Some people have tried doing visualization in TensorFlow.
See <https://github.com/insikk/Grad-CAM-tensorflow>.
However, TensorFlow has too much boilerplate, and in general I'm not familiar with it. I believe with the huge amount
of boilerplate around TensorFlow, figuring out the usage of existing visualization code on my particular models,
adapted to my particular needs, would possibly take more of my time than working on a pure PyTorch solution.

## Implementation

It's going to be implemented mainly through forward and backward hooks of `torch.nn.Module`. Since most of visualization
techniques focus on fiddling with ReLU layers,
this means that as long as your ReLU layers, as well as those layers which contain your interested units
are implemented using `torch.nn.Module`, not `torch.nn.functional`, then the code should work.

### Alternatives

While it's possible to define some modified ReLU layers,
as [suggested](https://discuss.pytorch.org/t/inherit-from-autograd-function/2117/2) by PyTorch developers,
this make break the code, as autograd assumes correct grad computation.
