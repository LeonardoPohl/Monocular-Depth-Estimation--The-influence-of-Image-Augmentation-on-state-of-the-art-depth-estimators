from transformers import GLPNFeatureExtractor, GLPNForDepthEstimation
import numpy as np
import torch

class GLPN:
  def __init__(self, kitti: bool):
    print("GLPN: Initialising...")
    print("GLPN: Saved Model not Found...")
    if kitti:
      self.feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-kitti")
      self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")
    else:
      self.feature_extractor = GLPNFeatureExtractor.from_pretrained("vinvino02/glpn-nyu")
      self.model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    print("GLPN: Model calculated ...")

  def predict_cpu(self, image):
    pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
      predicted_depth = self.model(pixel_values).predicted_depth
    
    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=pixel_values.shape[-2:],
                        mode="bicubic",
                        align_corners=False,
                )
                
    prediction = prediction.squeeze().cpu().numpy()
    
    return prediction
