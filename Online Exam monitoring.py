import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import glob
import tkinter as tk
from tkinter import messagebox




def start(video_path,screen_path):
    
    vid = cv2.VideoCapture(video_path)
    vid_screen = cv2.VideoCapture(screen_path)
    
    i=0
    while i <= 10:         #Skip starting frames 
        ret,frame = vid_screen.read()
        ret, img = vid.read()
        i = i+1

    screen_counter = 0
    flag = True
    r,c,d = frame.shape
    ref_screen = frame[80:130,550:int(c*3/4)]
    ref_screen = cv2.cvtColor(ref_screen,cv2.COLOR_BGR2GRAY)


    counter_screen_skip = 1
    crop1 = img
    k=0
    cx = 0
    cy = 0
    counter1 = 0
    cheat = 0
    diff = 0
    counter_face = 0
    print('Started')
    face = cv2.CascadeClassifier('/data/XML/Face_Classifier.xml') #Face Classifier
    eye = cv2.CascadeClassifier('/data/XML/eapplestroe/xml/Eye_Classifier.xml') #Eyes Classifier
    
    W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_path='/data/Result.mp4'    #video saves here
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (W, H))
    
    while k != 27:
        ret, img = vid.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) 
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5)) 
        windowClose = np.ones((5,5),np.uint8)
        windowOpen = np.ones((3,3),np.uint8)
        windowErode = np.ones((3,3),np.uint8)
        windowDilate = np.ones((2,2),np.uint8)
        cheating = np.str(cheat)
        cv2.putText(img,'Cheating = '+cheating,(25,25), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)


        ################################# Eyes Detection Start ##############################

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            crop = gray[y:y+h,x:x+w]
            crop_original = img[y:y+h,x:x+w]
            row,col = crop.shape
            crop_face = crop[0:int(row/2),0:col]
            eyes = eye.detectMultiScale(crop_face,scaleFactor=1.15, minNeighbors = 10)
            for (x1,y1,w1,h1) in eyes:
                center_x = int((x1+w1)/2)
                center_y = int((y1+h1)/2)

                cv2.rectangle(img, (x+x1, y+y1), (x+x1+w1, y+y1+h1),(255,0,0),2) 
                crop1 = crop[y1:y1+h1,x1:+x1+w1]
                r,c = crop1.shape
                center_x = int(c/2)
                center_y = int(r/2)
                crop1 = cv2.equalizeHist(crop1)
                crop_original1 = crop_original[y1+5:y1+h1-5,x1+5:+x1+w1-5]
                ret, thresh = cv2.threshold(crop1,30,255,cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, windowClose)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, windowErode)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, windowOpen)
                cnts = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                if len(cnts) >= 2:
                #find biggest blob
                    maxArea = 0
                    MAindex = 0     #to get the unwanted frame 
                    distanceX = [] #delete the left most (for right eye)
                    currentIndex = 0 
                    for cnt in cnts:
                        area = cv2.contourArea(cnt)
                        center = cv2.moments(cnt)
                        distanceX.append(cx)
                        if area > maxArea:
                            maxArea = area
                            MAindex = currentIndex
                        currentIndex = currentIndex + 1

                    del cnts[MAindex]    #remove the picture frame contour
                    del distanceX[MAindex]
                eye1 = 'right'
                if len(cnts) >= 2:#delete the left most blob for right eye
                    if eye1 == 'right':
                        edgeOfEye = distanceX.index(min(distanceX))
                    else:
                        edgeOfEye = distanceX.index(max(distanceX))
                        del contours[edgeOfEye]
                        del distanceX[edgeOfEye]
                if len(cnts) >= 1:    #get largest blob
                    maxArea = 0
                    for cnt in cnts:
                        area = cv2.contourArea(cnt)
                        if area > maxArea:
                            maxArea = area
                            largeBlob = cnt
                            if len(largeBlob) > 0:
                                center = cv2.moments(largeBlob)
                                if (center['m00'] != 0):
                                    cx,cy = int(center['m10']/center['m00']), int(center['m01']/center['m00'])
                    diff = np.sqrt(np.square(cx-center_x) + np.square(cy-center_y))
                cv2.circle(crop1,(cx,cy),2,255,-1)
                cv2.circle(crop1,(center_x,center_y),2,255,-1)

        if diff > 8:
            counter = counter+1
        elif diff < 8:
            counter = 0

        if counter == 12:
            cheat = cheat+1
        ################################# Eyes Detection End ##############################



        ########################### Face Recognition Start ###############################

        preds = []
        prev_pred = 0
        k=0

        crop1 = crop
        images = glob.glob('/data/Faces/*.jpg') #Put faces directory here
        matched = images[0]
        if counter_face % 50 == 0:
            for im in images:
                img2 = cv2.imread(im)
                gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                faces2 = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces2:
                    if(w<150):
                        continue
                    crop2 = img2[y:y+h-10,x+10:x+w-10]
                    crop2 = cv2.cvtColor(crop2,cv2.COLOR_BGR2GRAY)
                    crop2 = cv2.equalizeHist(crop2)
                    if w<70 and h<70:
                        continue
                    result = cv2.matchTemplate(crop1,crop2,cv2.TM_CCOEFF)
                    (_,pred,_,_) = cv2.minMaxLoc(result)
                    preds.append(pred)
                    if(pred > prev_pred):
                        matched = im
                        prev_pred = pred

            final_matched = ''
            index = len(matched)-7
            for i in range(4):
                temp = matched[index]
                final_matched = final_matched+temp
                index = index+1
        cv2.putText(img, 'Matched: '+final_matched, (x-30, y-15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
        counter_face = counter_face+1

        #################### Face Recognition End ###############################


        ##################### Screen Recording Start ############################

        
        ret,frame1 = vid_screen.read()
        if (type(frame1) is np.ndarray):
            if counter_screen_skip % 5 == 0:
                r1,c1,d1 = frame1.shape
                counter1 = np.str(screen_counter)
                cv2.putText(frame1,'Screen changed = '+counter1+' times',(35,35), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
                r1,c1,d1 = frame1.shape
                frame = frame1[80:130,550:int(c1*3/4)]
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                result = cv2.matchTemplate(frame,ref_screen,cv2.TM_CCOEFF)
                (_,pred,_,_) = cv2.minMaxLoc(result)

                if pred < 76000000 and flag == True:
                    screen_counter = screen_counter+1
                    flag = False
                elif pred > 76000000:
                    flag = True
                cv2.imshow('Screen Recording',frame1)
                vid_writer.write(frame1)

        counter_screen_skip =  counter_screen_skip+1


        ################### Screen recording End ##############################

        cv2.imshow('Cheating Detection',img)
        cv2.imshow('Thresholded',thresh)
        cv2.imshow('Eyes',crop1)
        k = cv2.waitKey(30)
    #print(preds)
    cv2.destroyAllWindows()
    vid_writer.release()

def click():
    if(_video.get() == '' or _screen.get() == ''):
        messagebox.showinfo('Error','Fill all the fields')
        return
    print(_video.get())
    btn1.configure(state = 'disable')
    start()


##################################### Main Starts #########################

def click():
    if(_video.get() == '' or _screen.get() == ''):
        messagebox.showinfo('Error','Fill all the fields')
        return
    print(_video.get())
    btn1.configure(state = 'disable')
    start(_video.get(),_screen.get())

win = tk.Tk()
win.resizable(10,10)
win.title('Exam Monitoring')
video = tk.Label(win,text = 'Video path')
video.grid(column=0,row=1)
screen = tk.Label(win,text = 'Screen path')
screen.grid(column=0,row=2)



_video = tk.StringVar()
_screen = tk.StringVar()
entry2 = tk.Entry(win,width=20,textvariable=_video)
entry2.grid(column=1,row=1)
entry2.focus()
entry3 = tk.Entry(win,width=20,textvariable=_screen)
entry3.grid(column=1,row=2)

btn1 = tk.Button(win,text = 'Start',command=click,width=20,foreground='white',highlightbackground = 'black')
btn1.grid(column=1,row=3)


win.mainloop()

##################################### Main Ends #########################







