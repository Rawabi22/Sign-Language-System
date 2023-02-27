from cvzone.ClassificationModule import Classifier
import numpy as np
import cv2
import os
import PIL
from PIL import ImageTk
import PIL.Image
import speech_recognition as sr
import pyttsx3
from itertools import count
import string
from tkinter import *
import math
try:
       import Tkinter as tk
except:
       import tkinter as tk
import numpy as np
import time
import customtkinter
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

image_x, image_y = 64,64
# from keras.models import load_model
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# def give_char():
#        import numpy as np
#        from keras.preprocessing import image
#        test_image = image.load_img('tmp1.png', target_size=(64, 64))
#        test_image = image.img_to_array(test_image)
#        test_image = np.expand_dims(test_image, axis = 0)
#        result = classifier.predict(test_image)
#        print(result)
#        chars="ABCDEFGHIJKMNOPQRSTUVWXYZ"
#        indx=  np.argmax(result[0])
#        print(indx)
#        return(chars[indx])


def button_function():
       cap = cv2.VideoCapture(0)
       detector = HandDetector(maxHands=2)
       classifier = Classifier("Model\keras_model.h5", "Model/labels.txt")
       offset = 20
       imgSize = 300
       # folder = "Data/C"
       counter = 0
       labels = ["R", "A", "W", "B", "I", 'my', 'name', 'yes', 'no', 'love you']
       # labels = ["A", "B", "C", "D", "E",'F', 'G', 'H', 'I', 'J','K','L','M','N','O','P','Q',"R",'S','T','U','V','W','X','Y','Z'] 
       # labels = ['R', 'A', 'W', 'B', 'I', 'my', 'name', 'thanks']
       while True:
              success, img = cap.read()
              imgOutput = img.copy()
              hands, img = detector.findHands(img)
              if hands:
                     hand = hands[0]
                     x, y, w, h = hand['bbox']
                     imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                     imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                     imgCropShape = imgCrop.shape
                     aspectRatio = h / w
                     if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            imgResizeShape = imgResize.shape
                            wGap = math.ceil((imgSize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize
                            prediction, index = classifier.getPrediction(
                            imgWhite, draw=False)
                            # print(prediction, index)
                     else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            imgResizeShape = imgResize.shape
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize
                            prediction, index = classifier.getPrediction(
                            imgWhite, draw=False)
                     cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                   (x + w+offset, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                     cv2.putText(imgOutput, labels[index], (x, y - 26),
                                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                     cv2.rectangle(imgOutput, (x-offset, y-offset),
                                   (x + w+offset, y + h+offset), (255, 0, 255), 4)
              # cv2.imshow("ImageCrop", imgCrop)
              # cv2.imshow("ImageWhite", imgWhite)
              cv2.imshow("Sign to Text", imgOutput)
              # cv2.waitKey(1)
              if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
def check_sim(i,file_map):
       for item in file_map:
              for word in file_map[item]:
                     if(i==word):
                            return 1,item
       return -1,""

op_dest = "filtered_data/"
alpha_dest = "alphabet/"
dirListing = os.listdir(op_dest)
editFiles = []
for item in dirListing:
       if ".webp" in item:
              editFiles.append(item)

file_map={}
for i in editFiles:
       tmp=i.replace(".webp","")
       #print(tmp)
       tmp=tmp.split()
       file_map[i]=tmp

def func(a):
       all_frames=[]
       final= PIL.Image.new('RGB', (380, 260))
       words=a.split()
       for i in words:
              flag,sim=check_sim(i,file_map)
              if(flag==-1):
                     for j in i:
                            print(j)
                            im = PIL.Image.open(alpha_dest+str(j).lower()+"_small.gif")
                            frameCnt = im.n_frames
                            for frame_cnt in range(frameCnt):
                                   im.seek(frame_cnt)
                                   im.save("tmp.png")
                                   img = cv2.imread("tmp.png")
                                   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                   img = cv2.resize(img, (380,260))
                                   im_arr = PIL.Image.fromarray(img)
                                   for itr in range(15):
                                          all_frames.append(im_arr)
              else:
                     print(sim)
                     im = PIL.Image.open(op_dest+sim)
                     im.info.pop('background', None)
                     im.save('tmp.gif', 'gif', save_all=True)
                     im = PIL.Image.open("tmp.gif")
                     frameCnt = im.n_frames
                     for frame_cnt in range(frameCnt):
                            im.seek(frame_cnt)
                            im.save("tmp.png")
                            img = cv2.imread("tmp.png")
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (380,260))
                            im_arr = PIL.Image.fromarray(img)
                            all_frames.append(im_arr)
       final.save("out.gif", save_all=True, append_images=all_frames, duration=100, loop=0)
       return all_frames      

img_counter = 0
img_text=''
class Tk_Manage(tk.Tk):
       def __init__(self, *args, **kwargs):     
              tk.Tk.__init__(self, *args, **kwargs)
              container = tk.Frame(self)
              container.pack(side="top", fill="both", expand = True)
              container.grid_rowconfigure(0, weight=1)
              container.grid_columnconfigure(0, weight=1)
              self.frames = {}
              for F in (StartPage, VtoS, TtoS):
                     frame = F(container, self)
                     self.frames[F] = frame
                     frame.grid(row=0, column=0, sticky="nsew")
              self.show_frame(StartPage)
       def show_frame(self, cont):
              frame = self.frames[cont]
              frame.tkraise()
class StartPage(tk.Frame):
       def __init__(self, parent, controller):
              tk.Frame.__init__(self,parent)
              load = PIL.Image.open("bg2.png")
              # load = load.resize((800, 750))
              render = ImageTk.PhotoImage(load)
              img = Label(self, image=render)
              img.image = render
              img.place(x=0, y=0)
              # label = tk.Label(self, text="Sign Langage Translator", font=("Verdana", 12))
              # label.pack(pady=10,padx=10)
              button1 = customtkinter.CTkButton(self, text="Sign to Text", command=button_function, text_color=("#ffffff"), font=("Verdana", 20), fg_color=("#FF914D"), bg_color=("#FF914D"), hover_color=("#FF914D"))
              button1.place(relx=0.5, rely=0.505, anchor=CENTER)
              button = customtkinter.CTkButton(self, text="Voice to Sign", command=lambda: controller.show_frame(VtoS),  text_color=("#ffffff"), font=("Verdana", 20), fg_color=("#FF914D"), bg_color=("#FF914D"), hover_color=("#FF914D"))
              button.place(relx=0.5, rely=0.615,  anchor=CENTER)
              button2 = customtkinter.CTkButton(self, text="Text to Sign", command=lambda: controller.show_frame(TtoS),  text_color=("#ffffff"), font=("Verdana", 20), fg_color=("#FF914D"), bg_color=("#FF914D"), hover_color=("#FF914D"))
              button2.place(relx=0.5, rely=0.725,  anchor=CENTER)            
class VtoS(tk.Frame):
       def __init__(self, parent, controller):
              cnt=0
              gif_frames=[]
              inputtxt=None
              tk.Frame.__init__(self, parent)
              load = PIL.Image.open("bg.png")
              render = ImageTk.PhotoImage(load)
              img = Label(self, image=render)
              img.image = render
              img.place(x=0, y=0)
              gif_box = tk.Label(self)
              message = tk.Label(self, text='', bg=('#ffffff'), font=("Verdana", 12))
              button1 = customtkinter.CTkButton(self, text="Back to Home", command=lambda: controller.show_frame(StartPage), text_color=("#ffffff"), font=("Verdana", 18), fg_color=("#FF914D"), hover_color=("#FF914D"))
              button1.place(relx=0.35, rely=0.95,  anchor=CENTER)
              
              def gif_stream():
                     global cnt
                     global gif_frames
                     if(cnt==len(gif_frames)):
                            return
                     img = gif_frames[cnt]
                     cnt+=1
                     imgtk = ImageTk.PhotoImage(image=img)
                     gif_box.imgtk = imgtk
                     gif_box.configure(image=imgtk)
                     gif_box.after(50, gif_stream)
              def start_record():
                     global MyText
                     message.place_forget()
                     gif_box.place_forget()
                     r = sr.Recognizer()
                     try:
                            with sr.Microphone() as source2:
                                   r.adjust_for_ambient_noise(source2, duration=3)
                                   audio2 = r.listen(source2)
                                   MyText = r.recognize_google(audio2)
                                   MyText = MyText.lower()
                                   message.config(text="Do you mean:\n"+MyText, fg="#FF8FA0")
                                   
                     except sr.RequestError as e:
                            print("Could not request results; {0}".format(e))
                            message.config(text="Could not request results; {0}".format(e), fg="#FF914D")
                     except sr.UnknownValueError:
                            print("unknown error occured")
                            message.config(text="Try again", fg="#FF914D")
                     message.place(relx=0.5, rely=0.68, anchor=CENTER)
              def output():
                     global gif_frames
                     global MyText
                     gif_frames = func(MyText)
                     global cnt
                     cnt = 0
                     gif_stream()
                     message.place_forget()
                     gif_box.place(relx=0.5, rely=0.68, anchor=CENTER)
              def clearb():
                     message.place_forget()
                     gif_box.place_forget()
              clearbtn = customtkinter.CTkButton(self, text="Clear", command=lambda: clearb(), text_color=("#ffffff"), font=("Verdana", 18), fg_color=("#FF914D"), hover_color=("#FF914D"))
              clearbtn.place(relx=0.65, rely=0.95,  anchor=CENTER)
              voice_button = customtkinter.CTkButton(self, text="Start Record", command=lambda: start_record(
              ), text_color=("#ffffff"), font=("Verdana", 20), fg_color=("#FF914D"), bg_color=("#FF914D"), hover_color=("#FF914D"))
              voice_button.place(relx=0.5, rely=0.2642,  anchor=CENTER)
              Display = customtkinter.CTkButton(self, text="Convert to Sign", command=lambda: output(
              ), text_color=("#ffffff"), font=("Verdana", 20), fg_color=("#FF914D"), bg_color=("#FF914D"), hover_color=("#FF914D"))
              Display.place(relx=0.5, rely=0.385,  anchor=CENTER)
              
class TtoS(tk.Frame):
       def __init__(self, parent, controller):
              tk.Frame.__init__(self, parent)
              load = PIL.Image.open("bg1.png")
              render = ImageTk.PhotoImage(load)
              img = Label(self, image=render)
              img.image = render
              img.place(x=0, y=0)
              cnt = 0
              gif_frames = []
              inputtxt = None
              gif_box = Label(self)
              button1 = customtkinter.CTkButton(self, text="Back to Home", command=lambda: controller.show_frame(StartPage), text_color=("#ffffff"), font=("Verdana", 18), fg_color=("#FF914D"), hover_color=("#FF914D"))
              button1.place(relx=0.35, rely=0.95,  anchor=CENTER)

              def gif_stream():
                     global cnt
                     global gif_frames
                     if (cnt == len(gif_frames)):
                            return
                     img = gif_frames[cnt]
                     cnt += 1
                     imgtk = ImageTk.PhotoImage(image=img)
                     gif_box.imgtk = imgtk
                     gif_box.configure(image=imgtk)
                     gif_box.after(50, gif_stream)
              def Take_input():
                     INPUT = inputtxt.get("1.0", "end-1c")
                     print(INPUT)
                     INPUT=INPUT.lower()
                     global gif_frames
                     gif_frames=func(INPUT)
                     global cnt
                     cnt=0
                     gif_stream()
                     gif_box.place(relx=0.5, rely=0.68, anchor=CENTER)

              def clearb():
                     gif_box.place_forget()
                     inputtxt.delete("1.0", "end")
              clearbtn = customtkinter.CTkButton(self, text="Clear", command=lambda: clearb(), text_color=(
              "#ffffff"), font=("Verdana", 18), fg_color=("#FF914D"), hover_color=("#FF914D"))
              clearbtn.place(relx=0.65, rely=0.95,  anchor=CENTER)
              inputtxt = tk.Text(self, height=3, width=50,bg="#F5F5F5", font=("Verdana", 12))
              inputtxt.place(relx=0.5, rely=0.24, anchor=CENTER)
              Display = customtkinter.CTkButton(self, text="Convert to Sign", command=lambda: Take_input(
              ), text_color=("#ffffff"), font=("Verdana", 20), fg_color=("#FF914D"), bg_color=("#FF914D"), hover_color=("#FF914D"))
              Display.place(relx=0.5, rely=0.385,  anchor=CENTER)

app = Tk_Manage()
app.geometry("600x600")
app.title('Sign Language')
app.iconbitmap("ILU2.ico")
app.mainloop()