import cv2
import os
import time
from skimage import io
from skimage.metrics import structural_similarity as ssim
from glob import glob
def create_saveDir(): #create "/save" directory
    try:
        path=os.getcwd()
        save='save'
        if not os.path.exists("/save"):
            path = os.path.join(path, save)
            os.makedirs(path)
            print('/save directory created. Keyframing results will be saved here.')
    except OSError:
        print(f"Results will be saved in /save ")
    return "save"
def create_dirFrames(path):#create "/save/videoname" directories
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")


def check_ifOneColor(img): #remove frame if it's a solid color (all black for example)
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    for x in range(height):
        for y in range(width):
            pv1= frame[x, y]#get the pixel value of the current pixel
            if(pv1!=frame[0,0]).any():
                return True #confirm that frame is not solid colour

    return False
def check_LightChange(img,img1):
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convert images to grayscale to compares pixels
    mframe = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    mframe = cv2.resize(mframe, dim, interpolation=cv2.INTER_AREA)

    for x in range(height):
        for y in range(width):
            if(abs(frame[x,y]-mframe[x,y])<=16):#test ken el far9 binet les pixel entre -32 et 32 (abs maaneha |-x|=x fel math
                return True
    return False
def checkStrucSim(savedir,idx,midx):
    if os.path.isfile(savedir+"/"+str(idx)+".jpg") and os.path.isfile(savedir+"/"+str(midx)+".jpg") and idx!=midx:
            image1_gray = io.imread(savedir+"/"+str(idx)+".jpg", as_gray=True)
            image2_gray = io.imread(savedir+"/"+str(midx)+".jpg", as_gray=True)
            ssi_index, _ = ssim(image1_gray, image2_gray, full=True, data_range=image2_gray.max() - image2_gray.min())# Compute the Structural Similarity Index (SSI)
            if(ssi_index>0.50):
                os.remove(savedir+"/"+str(idx)+".jpg")
                return False
    return True
def compareHist(img1,img2):#histogram comparison
    hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    if(metric_val<=0.42):
        #print(metric_val)
        return True

    return False
def compareHistDEL(img1,img2):#histogram comparison
    hist_img1 = cv2.calcHist([img1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([img2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    if(metric_val<=0.42):
        return True

    return False

def save_frame(video_path, save_dir):    #saving keyframs in /save/videoname
    name = video_path.split(".")[0].split("\\")[1]    #extract the video name without the extension
    save_path = os.path.join(save_dir, name)    #joining for example /projectYolo/save + name of the vid ducks = projectyolo/save/ducks
    create_dirFrames(save_path)    #creates /save/ducks
    cap = cv2.VideoCapture(video_path)    #gets the video from /videos/
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    #gets the total frame count
    print("total frame count:",length)
    fps = cap.get(cv2.CAP_PROP_FPS)    #extracts the fps of the video
    print('fps: ',fps)
    hop=3    #hop for videos 24+ frames
    if fps<24:
        hop=2
    ret, mainframe = cap.read()    #ret reutnrs false if no video (not 100% sure)
    frame=mainframe    #reads the first frame
    mainindex=0;
    if(check_ifOneColor(frame)):
        cv2.imwrite(f"{save_path}/{0}.jpg", mainframe)    #saves it if not monocolour
    idx = 1    #sets the index for next frame
    start = time.time()    #starting time to see how long it takes to process
    while True:
        if ret == False:    #releases the cap for if video not found
            cap.release()
            break
        if cap.get(cv2.CAP_PROP_FRAME_COUNT)<150: #save all frames if video has less than 150
            cv2.imwrite(f"{save_path}/{idx}.jpg", frame)
            continue
        if idx%hop==0:    #skipping 3 or 2 frames to reduce iterations
            if compareHist(mainframe, frame):    #calls compareHist
                if check_ifOneColor(frame):#check if one colour
                    if(check_LightChange(frame,mainframe)):#check if the frames are too similar
                        cv2.imwrite(f"{save_path}/{idx}.jpg", frame)    #if compare hist returns true it saves the frame
                        if(checkStrucSim(save_path,idx,mainindex)):
                            mainframe = frame    #changes the frame to compare to with the saved frame
                            mainindex=idx
        idx+=1#increase the index
        ret, frame = cap.read()#reads the next frame
    end = time.time()#ends the processing time for a video
    print('Full processing keyframes time for ',name,': ',end - start)
    return save_path#return the save path for redundancy check
if __name__ == '__main__':
    pathCWD = os.getcwd()#get the current path
    if os.path.exists("videos"):#test if videos directory exists
        print('Processing videos...')
        video_paths = glob("videos/*")#retrieve all the video paths
        save_dir=create_saveDir()#creates /save
        for path in video_paths:#iterates videos
            times=time.time()
            dirName=save_frame(path,save_dir)#dir name is the path for the directory of the extreacted frames /save/ducks exp
            dirName = os.path.join(pathCWD, dirName)#joins the current path + the extrraacted fame pathj
            print("time for "+path+" "+str(time.time()-times))
    else:
        print('Videos directory does not exist, make sure to place videos for keyframing in /videos.')