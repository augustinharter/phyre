import cv2
import numpy as np
import json

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
      for j in range((-(n_in-1)*stride), (n_out*stride)+1, stride):
        x, y = int(pos[10+j][0][0]), int(pos[10+j][0][1])
        frame = cv2.imread(f"{self.path}/{self.pos+i}/{10+j}.jpg")
        frame = cv2.copyMakeBorder(frame, padding,0,0,0, cv2.BORDER_CONSTANT, value=[255,255,255])
        frame = frame[-(y+r):-(y-r), x-r:x+r]
        cv2.imshow("Test", frame)
        cv2.waitKey(delay=visual_delay)
        if j > 0:
          # Red ball filter, somehow there are some red noise pixels, that need to be eroded
          frame = np.logical_and(frame[:,:,2] >195, frame[:,:,2] <205, frame[:,:,1]<3).astype(float)
          kernel = (5,5)
          frame = cv2.erode(frame, kernel)
          frame = cv2.dilate(frame, kernel)
          sub_y.append(frame)
        else:
          # Green ball filter
          frame = np.logical_and(frame[:,:,1] >195, frame[:,:,1] <205, frame[:,:,2]<3).astype(float)      
          sub_x.append(frame)
          if j == 0:
            # Static objects filter
            frame = cv2.imread(f"{self.path}/{self.pos+i}/{10+j}.jpg")
            frame = frame[-(y+r):-(y-r), x-r:x+r]
            frame = np.logical_or(frame.sum(axis=2)<5, np.logical_and(frame[:,:,0] >195, frame[:,:,1] <3, frame[:,:,2]<3)).astype(float)
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

if __name__ == "__main__":
  loader = Extractor("rollouts/test")
  X, Y = loader.extract(n=10, stride=3, n_in=3, visual_delay=500)
  print(X, Y)