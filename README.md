# Attention in Neural Networks

Base-architecture : ResNet 50   
Dataset : CIFAR-100 (https://www.cs.toronto.edu/~kriz/cifar.html)   
***
The key idea is to emphasize relevant information and suppress the rest.
- In neural entworks, information is compressed in the form of feature map.
- Feature map X Attention -> Refined feature map
  * ResNet50 stage 3 output feature map
  * Features are averaged over channel axis and normalized per layer statistics
  
Generalizable attention module
- Can be adapted to any convolutional neural networks
- Global Avg. Pooling -> Squeeze -> FC*2 (Hu et al., CVPR 2018)
- Increase a little learnable parameters
***
1. Implement attention module
2. Report the followings for each of it and compare them - Numbers to Report
  * Top-1 and Top-5 Errors   
    - of the baseline   
    - with channel attention   
  * Parameters and Flops   
3. Analyze the learned features using Grad-CAM method (http://arxiv.org/abs/1610.02391)
4. New createive Idea
