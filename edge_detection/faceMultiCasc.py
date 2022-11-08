import numpy as np
import cv2 
import hist_equal
import os

cv2CascPath = cv2.data.haarcascades
# frontFaceCasc = cv2.CascadeClassifier(cv2CascPath + 'haarcascade_frontalface_alt.xml')
frontFaceCasc = cv2.CascadeClassifier(cv2CascPath + 'haarcascade_frontalface_alt2.xml') 
# frontFaceCasc = cv2.CascadeClassifier(cv2CascPath + 'haarcascade_frontalface_default.xml')

profileFaceCasc= cv2.CascadeClassifier(cv2CascPath + 'haarcascade_profileface.xml')
eyesCasc = cv2.CascadeClassifier(cv2CascPath + 'haarcascade_eye.xml')
smileCasc= cv2.CascadeClassifier(cv2CascPath + 'haarcascade_smile.xml')
# leftEyeCasc= cv2.CascadeClassifier(cv2CascPath + 'haarcascade_lefteye_2splits.xml')
# rightEyeCasc= cv2.CascadeClassifier(cv2CascPath + 'haarcascade_righteye_2splits.xml')
 
cameraID = 0
localPath = os.path.dirname(os.path.realpath(__file__))
localPath = localPath.replace('\\','/')
localPath= str(localPath.replace( 'c:/', 'C:/')) + "/"
print(localPath)

bins = 256
it = 0
sampling = 1 
HistEq = 1
if sampling == 1:
    print("Sampling is On")
else:
    print("Sampling is Off")

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

capture = cv2.VideoCapture(cameraID)

while(True):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frontFaces = frontFaceCasc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    profFaces = profileFaceCasc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    eyesRes = eyesCasc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    smileRes = smileCasc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in frontFaces: 
        it = it + 1
        # print(it)
        fX = x + w
        fY = y + h
        roi_gray = gray[y:y+h, x:x+w]
        # roi_color = frame[y:y+h, x:x+w]  
        fColor = (255,0,0) #BGR
        stroke = 2 
        cv2.rectangle(frame, (x, y), (fX, fY), fColor, stroke )
        if (sampling == 1 and it < 10 ): 
            img_FG = localPath + "images/faces/myImgFG" + str(it) + ".png"
            # img_FC = "images/myImgFC" + str(it) + ".png"
            cv2.imwrite(img_FG, roi_gray)
            # cv2.imwrite(img_FC, roi_color)

        for (ex,ey,ew,eh) in eyesRes:
            eX = ex + ew
            eY = ey + eh
            if (ex > x and ey > y and eX < fX and eY < fY):
                roi_gray = gray[ey:ey+eh, ex:ex+ew]
                roi_color = frame[ey:ey+eh, ex:ex+ew]  
                eColor = (0,255,0) #BGR
                stroke = 2 
                cv2.rectangle(frame, (ex, ey), (eX, eY), eColor, stroke )
                if (sampling == 1 and it < 10 ): 
                    img_EG = localPath + "images/eyes/myImgEG" + str(it) + ".png"
                    cv2.imwrite(img_EG, roi_gray)
                                       
                    img_EC = localPath + "images/eyes/myImgEC" + str(it) + ".png"
                    cv2.imwrite(img_EC, roi_color)
                    if HistEq == 1:
                        
                        # roi_gray = hist_equal.histEqualizer(roi_gray, bins).astype('uint8') 
                        roi_gray = cv2.Canny(roi_gray,50, 160, apertureSize = 3)
                        # sobelx= cv2.Sobel(roi_gray,cv2.CV_64F,1,0,ksize=3)
                        # abs_sobelx= np.absolute(sobelx)
                        # sobely= cv2.Sobel(roi_gray,cv2.CV_64F,0,1,ksize=3)
                        # abs_sobely= np.absolute(sobely)
                        # roi_gray= abs_sobelx + abs_sobely
                        
                        # roi_gray = cv2.Canny(roi_gray,140, 160, apertureSize = 3)
                        kernel = np.ones((5,5),np.uint8)
                        # roi_gray = cv2.erode(roi_gray,kernel,iterations = 1)
                        # roi_gray = cv2.dilate(roi_gray,kernel,iterations = 2)
                        # roi_gray = cv2.erode(roi_gray,kernel,iterations = 1)
                        # roi_gray = cv2.dilate(roi_gray,kernel,iterations = 2) 
                        # roi_gray = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel)
                        # roi_gray = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)
                        # ret, roi_gray = cv2.threshold(roi_gray, 40, 210, cv2.THRESH_BINARY)
                        # roi_gray = cv2.erode(roi_gray,kernel,iterations = 1)
                        # roi_gray = cv2.dilate(roi_gray,kernel,iterations = 1)
                        # roi_gray = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel)
                        # roi_gray = cv2.morphologyEx(roi_gray, cv2.MORPH_CLOSE, kernel)

                        img_EGHE = localPath + "images/eyes/myImgEGHE" + str(it) + ".png"
                        cv2.imwrite(img_EGHE, roi_gray)
                        
                    
                for (sx,sy,sw,sh) in smileRes:
                    sX = sx + sw
                    sY = sy + sh
                    if (sx > x and sy > y and sX < fX and sY < fY and sY > eY):
                        roi_gray = gray[sy:sy+sh, sx:sx+sw]
                        # roi_color = frame[sy:sy+sh, sx:sx+sw] 
                        sColor = (0,255,255) #BGR
                        stroke = 2                       
                        cv2.rectangle(frame, (sx, sy), (sX, sY), sColor, stroke )
                        if (sampling == 1 and it < 10 ): 
                            img_SG = localPath + "images/mouth/myImgSG" + str(it) + ".png"
                            # img_SC = "images/myImgSC" + str(it) + ".png"
                            cv2.imwrite(img_SG, roi_gray)
                            # cv2.imwrite(img_SC, roi_color)

    for (x,y,w,h) in profFaces: 
        it = it + 1
        fX = x + w
        fY = y + h
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  
        fColor = (0,0,255) #BGR
        stroke = 2 
        cv2.rectangle(frame, (x, y), (fX, fY), fColor, stroke )

        for (ex,ey,ew,eh) in eyesRes:
            eX = ex + ew
            eY = ey + eh
            if (ex > x and ey > y and eX < fX and eY < fY):
                roi_gray = gray[ey:ey+eh, ex:ex+ew]
                roi_color = frame[ey:ey+eh, ex:ex+ew]  
                eColor = (0,255,0) #BGR
                stroke = 2 
                cv2.rectangle(frame, (ex, ey), (eX, eY), eColor, stroke )

                for (sx,sy,sw,sh) in smileRes:
                    sX = sx + sw
                    sY = sy + sh
                    if (sx > x and sy > y and sX < fX and sY < fY and sY > eY):
                        roi_gray = gray[sy:sy+sh, sx:sx+sw]
                        roi_color = frame[sy:sy+sh, sx:sx+sw] 
                        sColor = (255,255,0) #BGR
                        stroke = 2                       
                        cv2.rectangle(frame, (sx, sy), (sX, sY), sColor, stroke )


                        

    cv2.imshow('frame', frame)
  
    if cv2.waitKey(20) & 0xFF == ord('q'):
        print("Total iterations", it)
        break

capture.release()
cv2.destroyAllWindows()