import cv2
import sys
import numpy as np

def get_trans_mat():
    #src_pts = np.float32([[0, 33], [17, 0], [0, 290], [14, 319]])
    #dst_pts = np.float32([[0, 33], [70, 33], [0, 290], [70, 290]])
    src_pts = np.float32([[33, 0], [0, 17], [290, 0], [319, 14]])
    dst_pts = np.float32([[33, 0], [33, 70], [290, 0], [290, 70]])
    #src_pts = np.array([[0, 0], [320 , 0], [0, 110 ], [320 , 110]], dtype = "float32")
    #dst_pts = np.array([[0, 0], [320, 0], [320 * 0.35, 110], [320 * 0.65, 110]], dtype = "float32") 
    return cv2.getPerspectiveTransform(src_pts, dst_pts)
  
def view(binary):
    trans = get_trans_mat()
    height, width = binary.shape[:2]
    return cv2.warpPerspective(binary,trans, (width, height))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        exit()
    
    file_name = sys.argv[1]

    img = cv2.imread(file_name, 0)

    cv2.imshow("src image", img)
    M = get_trans_mat()
    pts = get_bird_view(img, M)
    cv2.imshow("pts image", pts)
    cv2.waitKey(5000)
    
