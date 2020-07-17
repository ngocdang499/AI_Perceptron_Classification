from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from pandas import DataFrame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import time, random 
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Perceptron_Visualization:
    def __init__(self,rnum,sleep):
        self.sleep = sleep

        self.x, self.y, self.centers = make_blobs(n_samples=rnum, centers=2, n_features=2, return_centers=True)

        # # scatter plotting the given data
        self.df = DataFrame(dict(x=self.x[:,0], y=self.x[:,1], label=self.y))

        self.fig, self.ax = plt.subplots()
        self.grouped = self.df.groupby('label')
        plt.plot()
        for key, group in self.grouped:

            c = (0.1+key*0.9/2, 0.50+0.5*key/2, 0.6+0.4*key/2)
            group.plot.scatter(x='x', y='y', color=c, label='Class '+str(key+1), ax=self.ax, s=70, edgecolors=(0.09, 0.4, 0.5))
        
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.y = self.y*2 - np.ones(self.y.shape)
        # draw classification boundary line
        # perceptron
        self.perceptron_line_x1 = np.array([0])
        self.perceptron_line_x2 = np.array([0])

        self.perceptron_line, = self.ax.plot(self.perceptron_line_x1, 
                           self.perceptron_line_x2, 
                           color='#3d3d3d', 
                           lw=3, 
                           zorder= 4, 
                           label='$perceptron$')

        plt.legend(loc='lower left', prop={'size': 10})
        # display epoches on the plot
        self.text = plt.text(0.1, 0.9, '$epoch:1$', horizontalalignment='center', verticalalignment='center', transform = self.ax.transAxes, fontsize=10)

        # basic formatting for the axes
        # plt.grid(True, linewidth=0.3)
        # plt.setp(self.ax.get_xticklabels(), visible=False)
        # plt.setp(self.ax.get_yticklabels(), visible=False)

        self.ax.set_xlabel('$x_{1}$', fontsize=10)
        self.ax.set_ylabel('$x_{2}$', fontsize=10) 

        self.ax.set_title('$Classification \ by \ Perceptron \ Algorithm$', fontsize=20)
        self.mark_point = self.ax.scatter(self.x[0], self.x[0], s=100, edgecolors='orange',facecolors='none', linewidths=2, marker='o')
        # draw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 

        plt.show(block = False)

    def perceptron_update_step(self, feature_vector, label, current_theta, current_theta_0):
        if label*(feature_vector@current_theta + current_theta_0) <= 0:
            current_theta += label*feature_vector
            current_theta_0 += label
        return current_theta, current_theta_0

    def perceptron(self, T):
        feature_matrix = self.x 
        labels = self.y
        # initializing theta, theta_0 value to use in different perceptron algorithms
        theta = np.ones(feature_matrix.shape[1])
        theta_0 = np.zeros(1)
    
        for t in range(T):
            for i in range(feature_matrix.shape[0]):
                theta, theta_0 = self.perceptron_update_step(feature_matrix[i], labels[i], theta, theta_0)
            
                self.mark_point.set_offsets(feature_matrix[i])

                # draw the classification boundary every epoch
                self.draw_perceptron_line(theta, theta_0,'perceptron')
                            
                # updating epoch 
                self.text.set_text(f'$epoch:{t+1}$')  
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.show(block=False)

                # delaying
                time.sleep(self.sleep)
        plt.show()
        return theta, theta_0

    def draw_perceptron_line(self,theta, theta_0, algorithm='perceptron'):
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
        self.perceptron_line.set_xdata(line_x1)
        self.perceptron_line.set_ydata(line_x2)
        return line_x1, line_x2