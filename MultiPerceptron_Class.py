from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import time, random 

class Multiclass_Perceptron_Visualization:
    def __init__(self, numPoints, numClass, sleep):
        # generate 2d classification dataset
        self.sleep = sleep
        self.numClass = numClass
        n=numClass
        self.X, self.y, self.centers = make_blobs(n_samples=numPoints, centers=n, n_features=2, return_centers=True)
        # print(self.centers)
        # scatter plot, dots colored by class value
        self.df = DataFrame(dict(x=self.X[:,0], y=self.X[:,1], label=self.y))

        self.fig, self.ax = plt.subplots()
        self.grouped = self.df.groupby('label')
        plt.plot()
        for key, group in self.grouped:
            c = (0.1+key*0.9/n, 0.50+0.5*key/n, 0.6+0.4*key/n)
            group.plot.scatter(x='x', y='y', color=c, label='Class '+str(key+1), ax=self.ax, s=70, edgecolors=(0.09, 0.4, 0.5))
        
        self.perceptron_line_x1 = np.zeros(self.numClass)
        self.perceptron_line_x2 = np.zeros(self.numClass)       
        
        for i in range(numClass):
            self.ax.scatter(self.centers[i,0], self.centers[i,1], s=100, c='green', marker='x')

        self.perceptron_line = list()
        for i in range(numClass):
            self.perceptron_line.append(self.ax.plot(self.perceptron_line_x1[i], 
                           self.perceptron_line_x2[i], 
                           color=(random.random(),random.random(),random.random()), 
                           lw=2,  
                           label=(f'$Classify \ {i+1} \ vs \ others$') ))
        
        self.ax.legend(loc='lower left', prop={'size': 10})
        # display epoches on the plot
        self.text = plt.text(0.1,0.9, '$epoch:1$',horizontalalignment='center', verticalalignment='center', transform = self.ax.transAxes, fontsize=10)

        self.ax.set_xlabel('$x_{1}$', fontsize=10)
        self.ax.set_ylabel('$x_{2}$', fontsize=10) 

        self.ax.set_title('$Classification \ by \ Perceptron \ Algorithm$', fontsize=15)
        self.markPoint = self.ax.scatter(self.X[0,0], self.X[0,1], s=100, edgecolors='orange',facecolors='none', linewidths=2, marker='o')
        # draw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 
        
        plt.show(block=False)

    def model(self, point, theta, theta_0):
        a =  np.dot(point, theta.T) + theta_0
        return a

    def perceptron_update_step(self, point, label, current_theta, current_theta_0):
        preVal = self.model(point, current_theta, current_theta_0)
        preLabel = np.argmax(preVal)
        # for i in range(self.numClass):
        if preLabel != label:
            current_theta[label] += point 
            current_theta_0[label] += 1

        for i in range(self.numClass):
            if preVal[i] >= 0 and i != label:
                current_theta[i] = current_theta[i] - point 
                current_theta_0[i] -= 1
        return current_theta, current_theta_0
        
    def perceptron(self, epoch):
        feature_matrix = self.X
        labels = self.y
        # initializing theta, theta_0 value to use in different perceptron algorithms
        theta = np.ones((self.numClass, feature_matrix.shape[1]))
        theta_0 = np.zeros(self.numClass)        
    
        for t in range(epoch):
            for i in range(feature_matrix.shape[0]):   
                # Calculate the predict function and update theta values
                theta, theta_0 = self.perceptron_update_step(feature_matrix[i], labels[i], theta, theta_0)                
                
                # Mark the current examine point
                self.markPoint.set_offsets(self.X[i])
                
                # draw the classification boundary every epoch
                for i in range(self.numClass):
                    self.draw_perceptron_line(theta[i], theta_0[i],i)
               
                # updating epoch 
                self.text.set_text(f'$epoch:{t+1}$')  
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.show(block=False)

                # delaying
                time.sleep(self.sleep)
        
        plt.show()
        return (theta, theta_0)

    def draw_perceptron_line(self,theta, theta_0, index):
        # generate the data points of the boundary line
        if  theta[1] != 0:
            line_x1 = np.linspace(-50,50,2)
            line_x2 = (-theta_0-(theta[0]*line_x1))/theta[1]
        elif theta[0] != 0:
            line_x2 = np.linspace(-50,50,2)
            line_x1 = (-theta_0-(theta[1]*line_x2))/theta[0]
        else:
            line_x1 = np.array([0])
            line_x2 = np.array([0])
    
    # updating the boundary line
        self.perceptron_line[index][0].set_xdata(line_x1)
        self.perceptron_line[index][0].set_ydata(line_x2)    
        return line_x1, line_x2