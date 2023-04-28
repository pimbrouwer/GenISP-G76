# GenISP-G76
**Paper reproducibility project**  
Pim Brouwer - 4670639  
Tim Geukers - 4589718  
The implementation shown in this GitHub is from scratch.  
## Introduction
Object detection is a well known task in the practice of computer vision. A lot of off-the-self object detectors exist nowadays with amazingly high accuracy. However, object detection in low-light conditions still remains problematic. 

This blog post tries to reproduce a paper that tries to solve this exact problem, detecting objects in low-light conditions. The authors Igor Morawski, Yu-An Chen, Yu-Sheng Lin, Shusil Dangi, Kai He, and Winston H. Hsu proposed a neural workflow in their paper “GenISP: Generalizing Image Signal Processing for Low-Light Object Detection”. Their goal is to find 12 parameters for the white balancing and colour correction of the image so that the object detector can have better results on dark images.

In this report, we try to reproduce their workflow by using the method mentioned in their paper, using the same dataset and thus trying to ensure the same result. Furthermore we will discuss the ease at which this paper was to reproduce.

## Method
Firstly, we downloaded the Nikon dataset which contains around 4000 images and the ground truth bounding box annotations of people, bicycles and cars. These images are taken in different low-light circumstances, differing between street lights and full dark pictures.

### Preprocessing
The paper starts with preprocessing dark images shot on a Nikon camera. These images are saved in the RAW format. This format contains the image data as intensity for every pixel and every colour channel. The main advantage of working with RAW files is that there is no preprocessing done yet by the camera, therefore these files are independent of camera type which results in more consistent results when used for object detection, even when different cameras are used. Each pixel contains a 2-by-2 array for different colour channels in the RGrGbB format. The colour channels are then packed so a single image now consists of 4 different channels so every channel is now its own array. The two green channels are averaged so the image is now in standard RGB format.  
![packing](https://github.com/pimbrouwer/GenISP-G76/blob/ea1e222925c641a4bc4f894f40d446377c4e87dc/image5.png)  
The next step in preprocessing is transforming the image for RGB colour space to the XYZ colour space. This is a device independent colour space that enables the model to generalise to unseen camera sensors. This transformation is done through multiplication of every pixel with the CST matrix which is included in the metadata of the image. The last step in preprocessing is resizing the image to 800 by 1333 resolution. 
We used rawpy to preprocess these images, where the exact implementation can be seen below. 
```
class preprocessor(object):
  def __call__(self, image):
    new_h = 800
    new_w = 1333


    raw = rawpy.imread(image)
    rgb = raw.postprocess()
    convert_tensor = transforms.ToTensor()
    resize = T.Resize((new_h,new_w), antialias=True)
    img = resize(convert_tensor(rgb))
    
    return img.requires_grad_(), new_h, new_w
```
After the images are preprocessed we can continue with the neural network. In the paper a diagram has been added depicting the workflow:  
![workflow](https://github.com/pimbrouwer/GenISP-G76/blob/ea1e222925c641a4bc4f894f40d446377c4e87dc/image2.png)  
### Whitebalance and colour correction
To edit the brightness and colours in the image, two networks are constructed. The image is first fed to the ConvWB network for whitebalancing. The resulting image is then fed to the ConvCC network for colour correction. The ConvWB and ConvCC steps are almost identical, but lead to a different amount of parameters. For the white balancing, 3 parameters (w11, w22 and w33) are needed. For the colour correction, 9 parameters (c11, c12, c13, c21, c22, c23, c31, c32 and c33) are needed. These parameters are applied to the 3 channels with the following matrix multiplication:  
![wb](https://github.com/pimbrouwer/GenISP-G76/blob/ea1e222925c641a4bc4f894f40d446377c4e87dc/image1.png)  
![cc](https://github.com/pimbrouwer/GenISP-G76/blob/ea1e222925c641a4bc4f894f40d446377c4e87dc/image4.png)  
  
The layers of these methods have been made with the earlier mentioned architecture. The ConvWB class is implemented with the following code:
```
class convWB(nn.Module):
  def __init__(self):
    super(convWB, self).__init__()


    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.leakyReLU = nn.LeakyReLU(0.1)
    self.maxPool1 = nn.MaxPool2d(kernel_size=7, stride=1)
    self.maxPool2 = nn.MaxPool2d(kernel_size=5, stride=1)
    self.maxPool3 = nn.MaxPool2d(kernel_size=3, stride=1)
    self.avgAdaptPool = nn.AdaptiveAvgPool2d(1)
    self.mlp = nn.Linear(128, 3)


  def forward(self, x):
    out = nn.functional.interpolate(input=x, size=256)
    out = self.conv1(out)
    out = self.leakyReLU(out)
    out = self.maxPool1(out)
    out = self.conv2(out)
    out = self.leakyReLU(out)
    out = self.maxPool2(out)
    out = self.conv3(out)
    out = self.leakyReLU(out)
    out = self.maxPool3(out)
    out = self.avgAdaptPool(out).flatten(start_dim=1, end_dim=-1)
    out = self.mlp(out)


    wb = x.clone()
    for i in range(len(out)):
      wb[i, 0, :, :] = x[i, 0, :, :] * out[i, 0]
      wb[i, 1, :, :] = x[i, 1, :, :] * out[i, 1]
      wb[i, 2, :, :] = x[i, 2, :, :] * out[i, 2]


    return wb
```
The code for ConvCC is very similar and is implemented like this:
```
class convCC(nn.Module):
  def __init__(self):
    super(convCC, self).__init__()


    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.leakyReLU = nn.LeakyReLU(0.1)
    self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=1)
    self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=1)
    self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=1)
    self.avgAdaptPool = nn.AdaptiveAvgPool2d(1)
    self.mlp = nn.Linear(128, 9)


  def forward(self, x):
    out = nn.functional.interpolate(input=x, size=256)
    out = self.conv1(out)
    out = self.leakyReLU(out)
    out = self.maxPool1(out)
    out = self.conv2(out)
    out = self.leakyReLU(out)
    out = self.maxPool2(out)
    out = self.conv3(out)
    out = self.leakyReLU(out)
    out = self.maxPool3(out)
    out = self.avgAdaptPool(out).flatten(start_dim=1, end_dim=-1)
    out = self.mlp(out)


    cc = x.clone()
    for i in range(len(out)):
      cc[i,0,:,:] = x[i,0,:,:]*out[i,0] + x[i,1,:,:]*out[i,1] + x[i,2,:,:]*out[i,2]
      cc[i,1,:,:] = x[i,0,:,:]*out[i,3] + x[i,1,:,:]*out[i,4] + x[i,2,:,:]*out[i,5]
      cc[i,2,:,:] = x[i,0,:,:]*out[i,6] + x[i,1,:,:]*out[i,7] + x[i,2,:,:]*out[i,8]


    return cc
```
### Shallow ConvNet
The Shallow ConvNet is then used to implement non-linearities and further enhance the image after the ConcCC step. This is implemented with the following code:
```
class convShallow(nn.Module):
  def __init__(self):
    super(convShallow, self).__init__()


    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=0)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)
    self.instNorm = nn.InstanceNorm2d(100)
    self.leakyReLU = nn.LeakyReLU(0.1)


  def forward(self, x):
    out = self.conv1(x)
    out = self.instNorm(out)
    out = self.leakyReLU(out)
    out = self.conv2(out)
    out = self.instNorm(out)
    out = self.leakyReLU(out)
    out = self.conv3(out)


    return out
```
### Detection
After the image has been modified it is time to compare the output to the object classifier. Since the authors also provided a ground truth file to compare the results of the object detector with the objects to be found. In this ground truth file there are bounding boxes for every object given in the COCO style annotation (x min, y min, width, height). Every bounding box also contains a label with the object to be detected. The labels in this dataset range are 1, 2 and 3 and correspond to person, bicycle and car respectively. 
We now use the off-the-shelf object detector resnet50 with fixed weights, which compares the found bounding boxes with the ground truth bounding boxes and calculates the classification loss (alpha-balanced focal loss) and regression loss (smooth L1-loss). Both these functions were already available in resnet50. The code for calculating the loss is:
```
for j in range(len(image_sh)):
      imageSingle = torch.unsqueeze(image_sh[j,:,:,:], 0)
      output = resnetModel(imageSingle, groundTruthData(json_file_path_train, img_name[j], img_scale_h[j], img_scale_w[j]))


      if loss_cls_batch == None: 
        loss_cls_batch = output['classification'].unsqueeze(0)
        loss_reg_batch = output['bbox_regression'].unsqueeze(0)
      else:
        loss_cls_batch = torch.cat((loss_cls_batch, output['classification'].unsqueeze(0)))
        loss_reg_batch = torch.cat((loss_reg_batch, output['bbox_regression'].unsqueeze(0)))
    
    
    loss_cls = torch.mean(loss_cls_batch)
    loss_reg = torch.mean(loss_reg_batch)
    loss = loss_cls + loss_reg
```
With all the forward passes performed and the loss calculated, a gradient can be calculated using PyTorch. With an optimizer the weights of the network can be adjusted to reduce the loss. This backpropagation is handled using prebuilt PyTorch functionalities. 

## Results
The authors had mentioned what kind of optimizer, epoch, batch-size and learning rate they have used. So we have used the same parameters. They used:
* Adam optimizer
* 15 epochs
* Batch-size of 8
* Initial learning rate of 0.01 which multiplies by 0.1 per 5 epochs.

With these parameters we trained the model on 100 images, because of this we changed the batch size to 10. Since the dataset took such a long time to upload it was unfortunately not realistically possible to upload the whole dataset. With these 100 images for training and 25 for testing we managed to get a 68% accuracy with the following loss curve:  
![loss](https://github.com/pimbrouwer/GenISP-G76/blob/ea1e222925c641a4bc4f894f40d446377c4e87dc/image3.png)  

## Conclusion
Because of the uploading problem we were unfortunately not able to fully reproduce the results in the paper. However, we do feel like we were able to reproduce the method and trained the model accordingly. All layers used in the paper where succesfully implemented and we were able to succesfully train the network. With a larger trainingset, our network would peform better and we might get the same results as the original authors of GenISP.

## Discussion
We conclude that the paper is reproducible, however, we did notice that it was quite tough to reproduce. A lot of obstacles were along the way since the paper only mentions the core principle of the method, not all the implementation problems you can encounter.

As mentioned, the results could have been better with more training. But nonetheless we are quite content with our results and see a lot of use cases for surveillance cameras or self driving vehicles since those often operate at night.

We enjoyed reproducing this paper and are quite happy to show that we managed to!

## Task divison
Pim Brouwer was resposible for the image loading and preprocessing, dataset managment and training. Tim Geukers was responsible for the ConvWB, ConvCC and ConvShallow implementation. For the rest of the project we worked closely together and thus are both responsible for all other work. 

