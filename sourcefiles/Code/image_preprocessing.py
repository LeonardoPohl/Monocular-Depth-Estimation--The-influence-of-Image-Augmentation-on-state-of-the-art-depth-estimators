from PIL import Image, ImageEnhance, ImageFilter

def nothing(image):
  return image

def greyscale(image):
  lst=[]
  for i in image.getdata():

      lst.append((int(i[0]*0.299+i[1]*0.587+i[2]*0.114), int(i[0]*0.299+i[1]*0.587+i[2]*0.114), int(i[0]*0.299+i[1]*0.587+i[2]*0.114))) ### Rec. 609-7 weights
      #lst.append(i[0]*0.2125+i[1]*0.7174+i[2]*0.0721) ### Rec. 709-6 weights

  image_grey = Image.new('RGB', image.size)
  image_grey.putdata(lst)
  #image_BW.save("greyscale_sample.png")

  return image_grey

def edge(image):
  image = greyscale(image).convert("RGB")
      
  # Calculating Edges using the passed laplican Kernel
  image_edge = image.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                          -1, -1, -1, -1), 1, 0))
  return image_edge

def edge_grey(image, threshold):
  foreground = edge(image).convert("RGBA") 
  datas = foreground.getdata()
  newData = [] 
  for item in datas: 
      if item[0] <= threshold and item[1] <= threshold and item[2] <= threshold: 
          newData.append((255, 255, 255, 0)) 
      else: 
          newData.append(item) 

  foreground.putdata(newData)

  image_edge_grey = greyscale(image)
  image_edge_grey.paste(foreground, (0, 0), foreground)
  return image_edge_grey

def edge_col(image, threshold):
  foreground = edge(image).convert("RGBA") 
  datas = foreground.getdata()
  newData = [] 
  for item in datas: 
      if item[0] <= threshold and item[1] <= threshold and item[2] <= threshold: 
          newData.append((255, 255, 255, 0)) 
      else: 
          newData.append(item) 

  foreground.putdata(newData)

  image_edge_colour = image.copy()
  image_edge_colour.paste(foreground, (0, 0), foreground)
  return image_edge_colour

def desaturated(image, desaturation):
  converter = ImageEnhance.Color(image)
  return converter.enhance(desaturation)
