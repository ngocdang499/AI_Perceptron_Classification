from Perceptron_Classification import Perceptron_Visualization

# Perceptron Visualization Controller

from tkinter import *
root= Tk() 
root.wm_title("Perceptron Visualization Controller")

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

lamdbaLb = Label(root, text="Lamda:")
lamdbaLb.pack()

lambdaInput = Entry(root, width=30)
lambdaInput.pack()
lambdaInput.insert(0,"")

def myClick():
    a = Perceptron_Visualization(int(pointInput.get()), float(sleepInput.get()), int(epochInput.get()), float(lambdaInput.get()))
    a.perceptron(a.x, a.y, int(epochInput.get()), float(lambdaInput.get()))

myBtn = Button(root, text="Update", command=myClick)
myBtn.pack()
root.mainloop()