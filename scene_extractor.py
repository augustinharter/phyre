import cv2
import numpy as np
import json
import pathlib
import numpy as np

class Extractor:
  """Extractor class to extract and save datasets from raw rollouts"""
  def __init__(self, path, num_frames = 40):
    super().__init__()
    self.path = path
    self.pos = 0
    self.num_frames = num_frames
  
  def rewind(self):
    """rewinding the Extractor if you want to use extract-method multiple times from same starting point"""
    self.pos = 0

  def extract(self, n=1, n_channels=3, stride=1, visual_delay=0, r=32):
    """method for extracting a dataset from a raw rollout"""
    padding = 20
    X = []
    Y = []
    center = self.num_frames//2
    for i in range(n):
      sub_x = []
      sub_y = []
      pos = json.load(open(f"{self.path}/{self.pos+i}/positions.txt"))
      # Filter for desired frames:
      x, y = int(pos[center][0][0]), int(pos[center][0][1])
      for j in range((-(n_channels-1)//2*stride), ((n_channels-1)//2*stride)+1, stride):
        img = cv2.imread(f"{self.path}/{self.pos+i}/{center+j}.jpg")
        img = cv2.copyMakeBorder(img, padding,0,0,0, cv2.BORDER_CONSTANT, value=[255,255,255])
        img = img[-(y+r):-(y-r), x-r:x+r]
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if visual_delay:
          cv2.imshow("Test", img)
          cv2.waitKey(delay=visual_delay)
      
        # Red ball filter, somehow there are some red noise pixels, that need to be eroded
        if j == 0:
          red = cv2.inRange(frame, np.array([0,245,190]), np.array([15,255,210]))
          sub_y.append(red)
      
        # Green ball filter
        green =  cv2.inRange(frame, np.array([50,245,190]), np.array([70,255,210]))
        sub_x.append(green)

        # Static objects filter
        if j == 0:
          #floor =  cv2.inRange(frame, np.array([110,245,190]), np.array([130,255,210]))
          static =  cv2.inRange(frame, np.array([0,0,0]), np.array([255,255,60]))

      # Appending Static in the end
      sub_x.append(static)

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
    """method for saving an extracted dataset"""
    X = np.array(X)
    Y = np.array(Y)
    #print(X, Y)
    for i in range(len(X)):
      pathlib.Path(f"{path}/{i}/train").mkdir(parents=True, exist_ok=True)
      for x in range(len(X[i])):
        cv2.imwrite(f"{path}/{i}/train/layer{x}.jpg", X[i][x])
      for y in range(len(Y[i])):
        cv2.imwrite(f"{path}/{i}/train/target{y}.jpg", Y[i][y])
        #cv2.imshow("test", Y[i][y])
        #cv2.waitKey(delay=500)

if __name__ == "__main__":
  loader = Extractor("rollouts/test")
  X, Y = loader.extract(n=100, stride=3, n_channels=5, visual_delay=0)
  loader.save(X, Y, "rollouts/test")