import cv2
import sys

#in_path = "/home/zz/code/FaceVeri/dgface/not_face.log"

def watch_img(in_path):
    with open(in_path, 'r') as f:
        try:
            while True:
                line = f.next().strip()
                I =  cv2.imread(line)
                cv2.imshow("face", I)
                cv2.waitKey(0)
        except StopIteration:
            pass

if __name__ == "__main__":
    in_path = sys.argv[1] 
    watch_img(in_path)
