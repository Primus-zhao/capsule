# CapsNet-PT

A PyTorch implementation of CapsNet based on Geoffrey Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
Capsule is a magic model, before implement it, i saw lots of pytorch implementation, but lots of them suffer from different mistakes, such as wrong softmax dim, wrong squash dim. However, all these models give acceptable results.:) 
Although all the models can be used in practice, Hinton's capsule model originates from Cognitive Neuroscience ,we should judge this model from a biological perpective instead of just checking its accuracy.

There is another very good youtube video for capsule explanation [Capsule Networks](https://www.youtube.com/watch?v=pPN8d0E3900), you can view it for better understanding.


## Steps
**Step 1.** 
Clone this repository with ``git``.

```
$ git clone https://github.com/Primus-zhao/CapsNet-PT.git
$ cd CapsNet-PT
```

**Step 2.** 
Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), ``mv`` and extract it into ``data/raw`` directory, then run python gen_pt.py to generate pt data for pytorch

```
$ wget -c -P data/raw http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
you can also download the data from lecun's homepage and put it in data folder, it's faster
$ gunzip data/raw/*.gz
the last step is for converting original binary file to pt file for pytorch
$ python gen_pt.py
```

**Step 3.** 
Start the training:
```
$ python main.py
```

The default parameters of batch size is 128, and epoch is 3, we will validate after 100 steps. You can also configure this with display_step variable in main.py. Be careful, since validation will go through 10000 test images, validation will be a bit slow, so don't set display_step too small

