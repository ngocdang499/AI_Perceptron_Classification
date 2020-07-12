import numpy as np
import random, time
import matplotlib.pyplot as plt
from matplotlib import animation as ani
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Perceptron_Visualization:
    def __init__(self,rnum,sleep,e,l):
        seedValue = random.randrange(pow(2,32)-1)
        np.random.seed(5)

        e = int(e)
        self.sleep = sleep

        self.x = np.random.normal(loc=1, scale=1, size=(int(rnum),2)) - np.array([0.5, 0])
        self.x[int(rnum/2):] = np.random.normal(loc=2, scale=1, size=(int(rnum/2),2)) + np.array([1.5,0])

        self.y = np.ones(int(rnum)) * -1
        self.y[int(rnum/2):] *= -1


        plt.style.use('seaborn-whitegrid')
        self.fig, self.ax = plt.subplots(1, figsize=(7, 7))

        # # scatter plotting the given data
        plt.plot(self.x[:int(self.x.shape[0]/2),0],self.x[:int(self.x.shape[0]/2),1], color='#97cbff', lw=0, marker='o', markersize=12)
        plt.plot(self.x[int(self.x.shape[0]/2):,0],self.x[int(self.x.shape[0]/2):,1], color='#ffc1e0', lw=0, marker='o', markersize=12)

        # draw classification boundary line
        # perceptron
        self.perceptron_line_x1 = np.array([0])
        self.perceptron_line_x2 = np.array([0])
        self.mark_point = self.ax.scatter(0, 0, s=300, c='black')

        self.perceptron_line, = self.ax.plot(self.perceptron_line_x1, 
                           self.perceptron_line_x2, 
                           color='#ff0080', 
                           lw=6, 
                           zorder= 4, 
                           label='$perceptron$')

        # average perceptron
        self.average_perceptron_line_x1 = np.array([0])
        self.average_perceptron_line_x2 = np.array([0])
        self.average_perceptron_line, = self.ax.plot(self.average_perceptron_line_x1, 
                                   self.average_perceptron_line_x2, 
                                   color='#ffd306', 
                                   lw=6, 
                                   zorder= 3, 
                                   label='$average \ perceptron$')

        # pegasos
        self.pegasos_line_x1 = np.array([0])
        self.pegasos_line_x2 = np.array([0])
        self.pegasos_line, = self.ax.plot(self.pegasos_line_x1, 
                        self.pegasos_line_x2, 
                        color='#8cea00', 
                        lw=6, 
                        zorder= 2, 
                        label='$pegasos$')

        # display epoches on the plot
        self.text = plt.text(4, 5, '$epoch:1$', fontsize=15)

        # basic formatting for the axes
        plt.grid(True, linewidth=0.3)
        plt.legend(loc='lower left', prop={'size': 15})
        plt.setp(self.ax.get_xticklabels(), visible=False)
        plt.setp(self.ax.get_yticklabels(), visible=False)

        self.ax.set_ylim(-3.5, 6.5)
        self.ax.set_xlim(-3, 7)

        self.ax.set_xlabel('$x_{1}$', fontsize=20)
        self.ax.set_ylabel('$x_{2}$', fontsize=20) 

        self.ax.set_title('$Classification \ by \ Perceptron \ Algorithm$', fontsize=20)

        # draw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 
        plt.show(block = False)

    def perceptron_update_step(self, feature_vector, label, current_theta, current_theta_0):
        if label*(feature_vector@current_theta + current_theta_0) <= 0:
            current_theta += label*feature_vector
            current_theta_0 += label
        return (current_theta, current_theta_0)

    def pegasos_update_step(self, feature_vector, label, L, eta, current_theta, current_theta_0):
        if label*(feature_vector@current_theta + current_theta_0) <= 1:
            current_theta =  (1 - eta*L)*current_theta + eta*label*feature_vector
            current_theta_0 = current_theta_0 + eta*label

        else:
            current_theta =  (1 - eta*L)*current_theta
        
        return (current_theta, current_theta_0)

    def perceptron(self,feature_matrix, labels, T, L):
        # initializing theta, theta_0 value to use in different perceptron algorithms
        theta = np.ones(feature_matrix.shape[1])
        theta_0 = np.zeros(1)
    
        sum_theta = np.zeros(feature_matrix.shape[1])
        sum_theta_0 = np.zeros(1)
    
        pegasos_theta = np.zeros(feature_matrix.shape[1])
        pegasos_theta_0 = np.zeros(1)
    
        update_counter = 0
        # np.random.shuffle(feature_matrix)
    
        for t in range(T):
            self.draw_perceptron_line(sum_theta/update_counter, sum_theta_0/update_counter,'average_perceptron')

            for i in range(feature_matrix.shape[0]):
                update_counter += 1
                eta = 1/np.sqrt(update_counter)
            
                theta, theta_0 = self.perceptron_update_step(feature_matrix[i], labels[i], theta, theta_0)
                sum_theta += theta
                sum_theta_0 += theta_0
            
                pegasos_theta, pegasos_theta_0 = self.pegasos_update_step(feature_matrix[i], labels[i], L, eta, pegasos_theta, pegasos_theta_0)

                self.mark_point.set_offsets(feature_matrix[i])

                # draw the classification boundary every epoch
                self.draw_perceptron_line(theta, theta_0,'perceptron')
                self.draw_perceptron_line(pegasos_theta, pegasos_theta_0,'pegasos')
                            
                # updating epoch 
                self.text.set_text(f'$epoch:{t+1}$')  
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.show(block=False)

                # delaying
                time.sleep(self.sleep)
        

            
        
        return (theta, theta_0, sum_theta, sum_theta_0, pegasos_theta, pegasos_theta_0)

    def draw_perceptron_line(self,theta, theta_0, algorithm='perceptron'):
        # generate the data points of the boundary line
        if  theta[1] != 0:
            line_x1 = np.linspace(-5,10,2)
            line_x2 = (-theta_0-(theta[0]*line_x1))/theta[1]
        elif theta[0] != 0:
            line_x2 = np.linspace(-5,10,2)
            line_x1 = (-theta_0-(theta[1]*line_x2))/theta[0]
        else:
            line_x1 = np.array([0])
            line_x2 = np.array([0])
    
    # updating the boundary line
        if algorithm == 'perceptron':
            self.perceptron_line.set_xdata(line_x1)
            self.perceptron_line.set_ydata(line_x2)
        elif algorithm == 'average_perceptron':
            self.average_perceptron_line.set_xdata(line_x1)
            self.average_perceptron_line.set_ydata(line_x2)
        elif algorithm == 'pegasos':
            self.pegasos_line.set_xdata(line_x1)
            self.pegasos_line.set_ydata(line_x2)
    
        return line_x1, line_x2


