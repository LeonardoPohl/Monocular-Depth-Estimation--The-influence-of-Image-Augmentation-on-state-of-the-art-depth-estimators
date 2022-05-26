from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import numpy as np
import torch
from PIL import Image
from networks.NewCRFDepth import NewCRFDepth
from torchvision import transforms

def _is_pil_image(img):
  return isinstance(img, Image.Image)


def _is_numpy_image(img):
  return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class NeWCFs:
  def __init__(self, kitti: bool):
    print("NeWCFs: Initialising...")
    print("NeWCFs: Saved Model not Found...")
    self.kitti = kitti

    if kitti:
      checkpoint_path = 'Code/checkpoints/model_kittieigen.ckpt'
    else:
      checkpoint_path = 'Code/checkpoints/model_nyu.ckpt'

    self.model = NewCRFDepth(version='large07', inv_depth=False, max_depth=10 if not kitti else 80)
    self.model = torch.nn.DataParallel(self.model)
    
    print("== Model Initialized")
    checkpoint = torch.load(checkpoint_path)
    self.model.load_state_dict(checkpoint['model'])
    self.model.eval()
    self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    self.transform = transforms.Compose([
                          ToTensor()
                      ])
    print("NeWCFs: Model calculated ...")

  def predict_cpu(self, image):
    if self.kitti:
      return self.predict_cpu_kitti(image)

    sample = self.transform({"image": np.asarray(image, dtype=np.float32) / 255.0})
    with torch.no_grad():
      pixel_values = torch.autograd.Variable(sample['image'])
      pred_depth = self.model(pixel_values)
    
    prediction = pred_depth.cpu().numpy().squeeze()

    return prediction

  def predict_cpu_kitti(self, image):
    sample = self.transform({"image": np.asarray(image, dtype=np.float32) / 255.0})
    
    with torch.no_grad():
      pixel_values = torch.autograd.Variable(sample['image'])
      pred_depth = self.model(sample['image'])
    
    prediction = pred_depth.cpu().numpy().squeeze()

    return prediction

  def to_tensor(self, pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
      raise TypeError(
          'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
    
    if isinstance(pic, np.ndarray):
      img = torch.from_numpy(pic.transpose((2, 0, 1)))
      return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

class ToTensor(object):
  def __init__(self, ):
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  
  def __call__(self, sample):
    image = sample['image']
    image = self.to_tensor(image)
    image = self.normalize(image)

    return {'image': image}

  def to_tensor(self, pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
      raise TypeError(
          'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
    
    if isinstance(pic, np.ndarray):
      img = torch.from_numpy(pic.transpose((2, 0, 1)))
      return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img