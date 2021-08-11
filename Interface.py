from tkinter import * 
import tkinter as tk
from Datasets import *


window=Tk()
window.title=('Mix and match Classification')
window.geometry('200x300')


def DSchoice():
    #load the choice variables here
    #Currently values won't transfer over to Datasets.py without error
    
    DSui = DSvar.get()
    
    if DSui==1: 
        DS='Mushroom'
       
    elif DSui==2:
        DS='Network'
    print(DS)
     

def FSchoice():
    FSui=FSvar.get()

    if FSui==3:
        FS='Kbest'
    elif FSui==4:    
        FS='Model'
    print(FS)
    #FSinput(FS)

def MLchoice():
    MLui=MLvar.get()

    if MLui==5:
        ML='Forest'
    elif MLui==6:
        ML='NeuralNet'
    print(ML)
    #MLinput(ML)

DS,FS,ML=''

DSvar=IntVar()
w=tk.Label(window, text="Choose your Data Set: ")
w.pack(fill=tk.X, pady=10)

R1=Radiobutton(window, text="Mushroom Edibility",variable=DSvar,value=1,command=DSchoice)
R1.pack(anchor=W,pady=5)

R2=Radiobutton(window, text="Network Intrusion",variable=DSvar,value=2,command=DSchoice)
R2.pack(anchor=W,pady=5)



FSvar=IntVar()
w=tk.Label(window, text="Choose your Feature Selection: ")
w.pack(fill=tk.X, pady=10)

R3=Radiobutton(window, text="K Best",variable=FSvar,value=3,command=FSchoice)
R3.pack(anchor=W,pady=5)

R4=Radiobutton(window, text="Linear Regression",variable=FSvar,value=4,command=FSchoice)
R4.pack(anchor=W,pady=5)

w=tk.Label(window, text="Choose your Classification Algorithm: ")
w.pack(fill=tk.X, pady=10)



MLvar=IntVar()
R5=Radiobutton(window, text="Random Forest",variable=MLvar,value=5,command=MLchoice)
R5.pack(anchor=W,pady=5)

R6=Radiobutton(window, text="Neural Network",variable=MLvar,value=6,command=MLchoice)
R6.pack(anchor=W,pady=5)


tempDS,tempFS,tempML=''

dataSetChoice(tempDS,tempFS,tempML)
#dataSetChoice(DSchoice(),FSchoice(),MLchoice())

label=Label(window)
label.pack()
window.mainloop()


