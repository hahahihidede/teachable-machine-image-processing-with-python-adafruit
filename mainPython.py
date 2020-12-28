#import neccesarry library
import numpy as np
import cv2
import tensorflow.keras as tf
import pyttsx3
import math
from Adafruit_IO import Client, Feed, RequestError

if __name__ == '__main__':
    #labels path
    labelsPath = "./model/labels.txt"
    #read label.txt each line
    labelsFile = open(labelsPath, 'r')

    #variable to store classes
    classes = []
    line = labelsFile.readline()
    #start loop
    while line:
        
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsFile.readline()
    
    labelsFile.close()

    #keras model path
    modelPath = './model/keras_model.h5'
    model = tf.models.load_model(modelPath, compile=False)

    #setup input image, in this case i use my embedded webcam as input device
    cap = cv2.VideoCapture(0)

    #setup frame width and height
    frameWidth = 1280
    frameHeight = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

    cap.set(cv2.CAP_PROP_GAIN, 0)
    
    #start loop
    while True:

        #set printpoint to true
        np.set_printoptions(suppress=True)

       #reshape input image
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

       
        check, frame = cap.read()
       
        #setup frame
        margin = int(((frameWidth-frameHeight)/2))
        squareFrame = frame[0:frameHeight, margin:margin + frameHeight]
       
        resizedImage = cv2.resize(squareFrame, (224, 224))
        
        modelImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)

        imageArray = np.asarray(modelImage)
        
        normalizedImageArray = (imageArray.astype(np.float32) / 127.0) - 1
       
        data[0] = normalizedImageArray

        #variable to store predictiosn
        predictions = model.predict(data)

        
        confThreshold = 90
        confidence = []
        confLabel = ""
        threshold_class = ""
       
        #create border in frame
        per_line = 2  
        borderedFrame = cv2.copyMakeBorder(
            squareFrame,
            top=0,
            bottom=30 + 15*math.ceil(len(classes)/per_line),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        #read classes and start predict
        for i in range(0, len(classes)):
           
            confidence.append(int(predictions[0][i]*100))
           
            if (i != 0 and not i % per_line):
                #put text in frame
                cv2.putText(
                    img=borderedFrame,
                    text=confLabel,
                    org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                confLabel = ""

            confLabel += classes[i] + ": " + str(confidence[i]) + "%; "
       
            if (i == (len(classes)-1)):
                cv2.putText(
                    img=borderedFrame,
                    text=confLabel,
                    org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                confLabel = ""
            #print something in some conditions
            if confidence[i] > confThreshold:
                threshold_class = classes[i]
                if classes [i]== "Unknown" :
                    print("TEU NYAHO")
                # if classes [i]== "Mangga Jelek" :
                #     print("TIDAK MANTAP")
          #put predicted classes in large font
        cv2.putText(img=borderedFrame,text=threshold_class,org=(int(0), int(frameHeight+20)),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,color=(255, 255, 255)
        )
        #show frame
        cv2.imshow("Frame", borderedFrame)
        k = cv2.waitKey(10) & 0xff 
        if k == 27:
            break


#https://github.com/hahahihidede/
