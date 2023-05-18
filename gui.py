import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras_facenet import FaceNet
import pandas as pd


#Loading Necessory models and files
df=pd.read_csv('classes.csv')
NAMES_LIST = list(df['Names'])
nn_model = load_model('face_model.h5',compile=False)
facenet = FaceNet()
haarcascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # using this bcz MTCNN is very slow

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Feed")
        
        self.FaceRecognise = False 
        self.PoseDetect = False 

        # Create camera port selection dropdown
        self.port_var = tk.IntVar()
        self.port_label = tk.Label(root, text="Select camera port:")
        self.port_label.pack()
        self.port_dropdown = tk.OptionMenu(root, self.port_var, *self.get_available_ports())
        self.port_dropdown.pack()
        
        # Create canvas to display camera feed
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()
        
        # Checkbox for face recognition
        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(root, text="Enable Face Recognition", variable=self.checkbox_var,
                                       command=self.toggle_face_recognition)
        self.checkbox.pack()
        
        # Checkbox for pose detection
        self.checkbox_var2 = tk.BooleanVar()
        self.checkbox2 = tk.Checkbutton(root, text="Enable Pose Detection", variable=self.checkbox_var2,
                                       command=self.toggle_pose_detection)
        self.checkbox2.pack()
        
        # Button to start the camera feed
        self.start_button = tk.Button(root, text="Start", command=self.start_camera)
        self.start_button.pack()
        
        # Button to stop the camera feed
        self.stop_button = tk.Button(root, text="Stop", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack()
        
        self.cap = None
        self.is_camera_active = False
    
    def get_available_ports(self):
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr
    
    def toggle_face_recognition(self):
        self.FaceRecognise = self.checkbox_var.get()
    def toggle_pose_detection(self):
        self.PoseDetect = self.checkbox_var2.get()
    
    def start_camera(self):
        if not self.is_camera_active:
            port = int(self.port_var.get())
            print(port)
            self.cap = cv2.VideoCapture(port)
            self.is_camera_active = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.show_camera_feed()
    
    def show_camera_feed(self):
        if self.is_camera_active:
            ret, frame = self.cap.read()
            if ret:
                if self.FaceRecognise:
                    frame=self.Recognise(frame)
                if self.PoseDetect:
                    #frame = call pose detection module
                    pass
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, image=image, anchor=tk.NW)
                self.canvas.image = image  # Store the reference to prevent image garbage collection
            
            self.canvas.after(60, self.show_camera_feed)
    
    def stop_camera(self):
        if self.is_camera_active:
            self.cap.release()
            self.is_camera_active = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    def Recognise(self,input_frame):
        rgb_img = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x,y,w,h in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv2.resize(img, (160,160)) # 1x160x160x3
            img = np.expand_dims(img,axis=0)
            ypred = facenet.embeddings(img)
            probs=nn_model.predict(ypred)
            class_index = np.argmax(probs) # gives me max probabilit's index value
            if probs[0][class_index]>0.85: # Display prediction only if prob > 0.85
                cv2.rectangle(input_frame, (x,y), (x+w,y+h), (255,0,255), 5)
                cv2.putText(input_frame, str(NAMES_LIST[class_index]), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(input_frame, (x,y), (x+w,y+h), (255,0,255), 5)
                cv2.putText(input_frame, str("Unknown"), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 3, cv2.LINE_AA)
        return input_frame

root = tk.Tk()
app = CameraApp(root)
root.mainloop()
