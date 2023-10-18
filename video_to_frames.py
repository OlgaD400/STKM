import cv2 

def FrameCapture(path):
  
    # Path to video file
    vidObj = cv2.VideoCapture(path)
  
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
  
        # Saves the frames with frame-count
        cv2.imwrite("cv_data/plume/fixed_avg/frame%d.jpg" % count, image)
  
        count += 1
  
        
FrameCapture('cv_data/plume/fixed_avg.mp4')