import cv2
import numpy as np
import json
import pathlib
import numpy as np

class Extractor:
  def __init__(self, path):
    super().__init__()
    self.path = path
    self.pos = 0
  
  def rewind(self):
    self.pos = 0

  def extract(self, n=1, n_in=3, n_out=3, stride=1, visual_delay=0, r=32):
    padding = 20
    X = []
    Y = []
    for i in range(n):
      sub_x = []
      sub_y = []
      pos = json.load(open(f"{self.path}/{self.pos+i}/positions.txt"))
      # Filter for desired frames:
      x, y = int(pos[10][0][0]), int(pos[10][0][1])
      for j in range((-(n_in-1)*stride), (n_out*stride)+1, stride):
        img = cv2.imread(f"{self.path}/{self.pos+i}/{10+j}.jpg")
        img = cv2.copyMakeBorder(img, padding,0,0,0, cv2.BORDER_CONSTANT, value=[255,255,255])
        img = img[-(y+r):-(y-r), x-r:x+r]
        if visual_delay:
          cv2.imshow("Test", img)
          cv2.waitKey(delay=visual_delay)
        if j > 0:
          # Red ball filter, somehow there are some red noise pixels, that need to be eroded
          frame = np.logical_and(img[:,:,2] >195, img[:,:,2] <205, img[:,:,1]<3).astype(float)
          kernel = (5,5)
          frame = cv2.erode(frame, kernel)
          frame = cv2.dilate(frame, kernel)
          sub_y.append(frame)
        else:
          # Green ball filter
          frame = np.logical_and(img[:,:,1] >195, img[:,:,1] <205, img[:,:,2]<3).astype(float)      
          sub_x.append(frame)
          # Static objects filter
          if j == 0:
            frame = np.logical_or(img.sum(axis=2)<5, np.logical_and(img[:,:,0] >195, img[:,:,1] <3, img[:,:,2]<3)).astype(float)
            sub_x.append(frame)
        
      if visual_delay:
        for x in sub_x:
          cv2.imshow("Input", x)
          cv2.waitKey(delay=visual_delay)
        
        for y in sub_y:
          cv2.imshow("Target", y)
          cv2.waitKey(delay=visual_delay)

      X.append(sub_x)
      Y.append(sub_y)

    self.pos += n
    return X, Y
  
  def save(self, X, Y, path):
    X = np.array(X)
    Y = np.array(Y)
    for i in range(len(X)):
      pathlib.Path(f"{path}/{i}/train").mkdir(parents=True, exist_ok=True)
      for x in range(len(X[i])):
        cv2.imwrite(f"{path}/{i}/train/layer{x}.jpg", X[i][x]*255)
      for y in range(len(Y[i])):
        cv2.imwrite(f"{path}/{i}/train/target{y}.jpg", Y[i][y]*255)
        cv2.imshow("test", Y[i][y])
        #cv2.waitKey(delay=500)

if __name__ == "__main__":
  loader = Extractor("rollouts/test")
  X, Y = loader.extract(n=99, stride=3, n_in=3, visual_delay=0)
  loader.save(X, Y, "rollouts/test")