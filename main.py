from Perceptron_Classification import Perceptron_Visualization
from MultiPerceptron_Class import Multiclass_Perceptron_Visualization

# Perceptron Visualization Controller

from tkinter import *
root= Tk() 
root.wm_title("Perceptron Visualization Controller")

classLb = Label(root, text="Number of Class:")
classLb.pack()

classInput = Entry(root, width=30)
classInput.pack()
classInput.insert(0,"")

pointLb = Label(root, text="Number of Points:")
pointLb.pack()

pointInput = Entry(root, width=30)
pointInput.pack()
pointInput.insert(0,"")

sleepLb = Label(root, text="Pause time (second):")
sleepLb.pack()

sleepInput = Entry(root, width=30)
sleepInput.pack()
sleepInput.insert(0,"")

epochLb = Label(root, text="Epoch:")
epochLb.pack()

epochInput = Entry(root, width=30)
epochInput.pack()
epochInput.insert(0,"")

def myClick():  
    if int(classInput.get()) > 2:
        a = Multiclass_Perceptron_Visualization(int(pointInput.get()),int(classInput.get()),float(sleepInput.get()))
        a.perceptron(int(epochInput.get()))      
    elif int(classInput.get()) == 2:
        a = Perceptron_Visualization(int(pointInput.get()), float(sleepInput.get()))
        a.perceptron(int(epochInput.get()))

myBtn = Button(root, text="Update", command=myClick)
myBtn.pack()
root.mainloop()