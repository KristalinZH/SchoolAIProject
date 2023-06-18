import tensorflow as tf
from keras import layers, models
import numpy as np
from PIL import Image
import tkinter as tk
import customtkinter as ctk
import os

def takeDirectoryLable(dir:str)->int:
    if dir=="Asteroids":
        return 1
    if dir=="Galaxies":
        return 2
    if dir=="Nebulas":
        return 3
    if dir=="Planets":
        return 4
    if dir=="Stars":
        return 5
    return 6

def normalizeImage(img:Image)->np.ndarray:
    image_array=np.array(img)
    normalized_array=image_array/255.0
    return normalized_array

def getData(typeOfData:str)->tuple[list,list]:
    images=[]
    labels=[]
    dataset_path=f'{os.getcwd()}\Dataset'
    dirs=os.listdir(dataset_path)
    for dir in dirs:
        for _,_,files in os.walk(f'{dataset_path}\{dir}\{typeOfData}'):
            for file in files:
                img=Image.open(f'{dataset_path}\{dir}\{typeOfData}\{file}')
                img=img.convert('RGB')               
                img=img.resize((128,128))
                img=normalizeImage(img)                              
                images.append(img)
                labels.append(takeDirectoryLable(dir))
    return (images,labels)

def createNeuralNetwork()->models.Sequential:
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation='leaky_relu',input_shape=(128,128,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='leaky_relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.8))
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(6,activation='softmax'))
    return model

def encodeLabels(labels:list)->np.ndarray:
    encodedLabels=np.zeros((len(labels),6))
    for i,label in enumerate(labels):
        encodedLabels[i, label-1] = 1
    return encodedLabels

def createModel()->models.Sequential:
    train_data_labels=getData('Train')
    test_data_labels=getData('Test')
    train_data=np.array(train_data_labels[0])
    train_labels=encodeLabels(train_data_labels[1])
    test_data=np.array(test_data_labels[0])
    test_labels=encodeLabels(test_data_labels[1])

    model=createNeuralNetwork()

    opt = tf.keras.optimizers.Adam(learning_rate=0.0008)
    model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    
    history = model.fit(train_data, train_labels, epochs=30, 
                    validation_data=(test_data, test_labels))
    
    result = model.evaluate(test_data,  test_labels, verbose=2)
    print(result)
    model.save('NewModel.h5')
    return model

def loadModel(pathToModel:str)->models.Sequential:
    model:models.Sequential=None
    if os.path.isfile(pathToModel):
        model=models.load_model(pathToModel)
    else:
        model=createModel()
    return model

def convertImageForGuessing(imgPath:str)->np.ndarray:
    imgToGuess=Image.open(imgPath)
    imgToGuess=imgToGuess.convert('RGB')
    imgToGuess=imgToGuess.resize((128,128))
    normalized_image_array=normalizeImage(imgToGuess)
    normalized_image_array=[normalized_image_array]
    normalized_image_array=np.array(normalized_image_array)
    return normalized_image_array

def uploadAction():
    global globalImagePath
    imgPath = ctk.filedialog.askopenfilename()
    if imgPath:
        globalImagePath = imgPath
        img = ctk.CTkImage(dark_image=Image.open(imgPath), size=(300, 200))
        ctk.CTkLabel(app, image=img, text="").place(x=450, y=150)

def guessWhatIsTheImage(model:models.Sequential,imgPath:str)->str:
    categories={1:"Asteroid",2:"Galaxy",3:"Nebula",4:"Planet",5:"Star",6:"Supernova"}
    ndarray_image=convertImageForGuessing(imgPath)
    categoryIndex=np.argmax(model.predict(ndarray_image))+1
    return f"This is a/an {categories[categoryIndex]}"

def displayAnswer():
    global answeLabel
    if globalImagePath:
        if answeLabel:
            answeLabel.destroy()
        answeLabel=ctk.CTkLabel(master=app, text=guessWhatIsTheImage(model,globalImagePath), font=("Times New Roman", 30), fg_color="black")
        answeLabel.place(x=475,y=400)
    
if __name__=='__main__':
    global globalImagePath
    globalImagePath = None
    guessResult=None
    global answeLabel
    answeLabel:ctk.CTkLabel=None
    model=loadModel(f"{os.getcwd()}\\SpaceObjectAIModel.h5")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.resizable(False, False)
    app.geometry("1170x651")  # 1300x600
    app.title("Space Objects Recogniser")

    backgroundImage = tk.PhotoImage(file="background.png")
    background = tk.Label(app, image=backgroundImage)
    background.place(x=0, y=0, relwidth=1, relheight=1)

    ctk.CTkLabel(master=app, text="AI Space Objects Recogniser", font=("Times New Roman", 30), fg_color="black").place(x=430,
                                                                                                            y=80)

    ctk.CTkButton(master=app, text="Import", bg_color="black", fg_color="black", font=("Times New Roman", 20),
          command=uploadAction).place(x=440, y=550)
 
    ctk.CTkButton(master=app, text="Generate Answer", font=("Times New Roman", 20), fg_color="black",
          command=displayAnswer).place(x=630, y=550)

    app.mainloop()
    

    
    
    