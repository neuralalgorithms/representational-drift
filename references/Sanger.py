#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Reference:
# from website: http://jeremy.fix.free.fr/Softwares/neural_pca.html 

# Simulation of the Generalized Hebbien Learning rule
# http://dspace.mit.edu/bitstream/handle/1721.1/6976/AITR-1086.pdf?sequence=2
import numpy 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to generate samples according to a gaussian distribution
# _________________________________________________________________

mu1 = 0.0
std1 = 1.0
mu2 = 0.0
std2 = 0.25
orientation = numpy.random.random() * numpy.pi
nb_samples = 400

def gauss_distrib(theta, mu1, std1, mu2, std2, nb_samples):
    samples = numpy.zeros((nb_samples,2));
    for i in range(nb_samples):
        u1 = numpy.random.random();
        u2 = numpy.random.random();
        T1 = numpy.sqrt(-2 * numpy.log(u1)) * numpy.cos(2*numpy.pi*u2);
        T2 = numpy.sqrt(-2 * numpy.log(u1)) * numpy.sin(2*numpy.pi*u2);
        
        samples[i,0] = mu1 +  (std1*T1 * numpy.cos(theta) - std2*T2 * numpy.sin(theta));
        samples[i,1] = mu2 +  (std1*T1 * numpy.sin(theta) + std2*T2 * numpy.cos(theta));	
    return samples

# We precompute the samples
samples = gauss_distrib(orientation, mu1,std1,mu2,std2,nb_samples)

# Center the data
# _______________________________

samples[:,0] -= numpy.mean(samples[:,0])
samples[:,1] -=numpy.mean(samples[:,1])

# Functions to update the network
# _______________________________

# 2 inputs, 2 outputs
weights = 0.1*(numpy.random.random((2,2)))

# For keeping track of the weights
weights_history = weights.copy().reshape((4,1))

initial_lrate = 0.5
lrate_history = numpy.array([initial_lrate])
presented_example = numpy.zeros((2,1))

def update_data():
    global weights, presented_example, weights_history, lrate_history
    
    # We take a random sample
    index_sample = numpy.random.randint(0,nb_samples)
    
    presented_example[0] = samples[index_sample,0]
    presented_example[1] = samples[index_sample,1]
    output = numpy.zeros((2,1))
    
    # We compute the output from this input
    for j in range(2):
        for i in range(2):
            output[j] += weights[i][j] * samples[index_sample,i]
    
    # Update the weights with the Sanger's rule
    oldweights = weights.copy()
    
    for j in range(2):
        for i in range(2):
            weights[i][j] = oldweights[i][j] + update_data.lrate * ( samples[index_sample,i]*output[j])
            for k in range(0,j+1):
                weights[i][j] -= update_data.lrate * output[j]* oldweights[i][k] * output[k]

    # Keep track of the evolution of the weights and learning rate, for display purpose only
    if(weights_history.shape[1] >= update_data.maxepoch):
        weights_history = numpy.hstack((weights_history[:,1:],weights.reshape((4,1))))
        lrate_history = numpy.append(lrate_history[1:], [update_data.lrate])
    else:
        weights_history = numpy.hstack((weights_history,weights.reshape((4,1))))
        lrate_history = numpy.append(lrate_history, [update_data.lrate])

    update_data.epoch = update_data.epoch + 1
    
    # Update the learning rate
    update_data.lrate = update_data.initial_lrate / (float(update_data.epoch)/nb_samples+1.0)


update_data.epoch = 1
update_data.maxepoch = nb_samples * 10 # 10 presentations of all the samples
update_data.initial_lrate = initial_lrate
update_data.lrate = update_data.initial_lrate

# Functions to deal with the display
# __________________________________

fig = plt.figure(facecolor='white')
ax_samples = fig.add_subplot(211)
plt.xlabel('X position')
plt.ylabel('Y position')

ax_weights = fig.add_subplot(212)
plt.xlabel('Epoch')
plt.ylabel('Weight value')

ax_samples.plot(samples[:,0], samples[:,1],'o', markersize=7, label='Samples')
# 1st component
line0, = ax_samples.plot([0.0, weights[0][0]], [0.0, weights[0][1]], color='#00FF00', linewidth = 3, label='1st component')
# 2nd component
line1, = ax_samples.plot([0.0, weights[1][0]], [0.0, weights[1][1]], '--',color='#00FF00', linewidth = 3, label='2nd component')
# Draw a circle of unitary radius for checking the norms of the weights
ax_samples.plot(numpy.cos(numpy.linspace(0,2*numpy.pi,100)), numpy.sin(numpy.linspace(0,2*numpy.pi,100)),'k--');
# Show the currently processed sample in red
selected_sample, = ax_samples.plot([presented_example[0]], [presented_example[1]], 'or', markersize = 7, label='Current sample')

ax_samples.set_xlim(-max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))), max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))))
ax_samples.set_ylim(-max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))), max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))))
ax_samples.set_aspect('equal')
ax_samples.legend(loc=(1.2,0.5))

# Manage the display of the weights and learning rates
plot_weights_history0, = ax_weights.plot([0.0], weights_history[0,:], 'b', label='1st component')
plot_weights_history1, = ax_weights.plot([0.0], weights_history[1,:], 'b')
plot_weights_history2, = ax_weights.plot([0.0], weights_history[2,:], 'g', label='2nd component')
plot_weights_history3, = ax_weights.plot([0.0], weights_history[3,:], 'g')
plot_lrate_history, = ax_weights.plot([0.0], lrate_history, 'r')
ax_weights.set_xlim(0, update_data.maxepoch)
ax_weights.set_ylim(-1.5, 1.5)
ax_weights.legend()


def init_draw():
    line0.set_data([],[])
    line1.set_data([],[])
    selected_sample.set_data([],[])
    plot_weights_history0.set_data([],[])
    plot_weights_history1.set_data([],[])
    plot_weights_history2.set_data([],[])
    plot_weights_history3.set_data([],[])
    plot_lrate_history.set_data([],[])
    return line0, line1, selected_sample, plot_weights_history0, plot_weights_history1, plot_lrate_history    

def updatefig(i):

    update_data()

    # Update the data to display
    line0.set_data([0.0, weights[0][0]], [0.0, weights[1][0]])
    line1.set_data([0.0, weights[0][1]], [0.0, weights[1][1]])

    selected_sample.set_data([presented_example[0]], [presented_example[1]])

    plot_weights_history0.set_xdata(numpy.arange(0,numpy.size(weights_history[0,:])))
    plot_weights_history0.set_ydata(weights_history[0,:])

    plot_weights_history1.set_xdata(numpy.arange(0,numpy.size(weights_history[1,:])))
    plot_weights_history1.set_ydata(weights_history[1,:])

    plot_weights_history2.set_xdata(numpy.arange(0,numpy.size(weights_history[2,:])))
    plot_weights_history2.set_ydata(weights_history[2,:])

    plot_weights_history3.set_xdata(numpy.arange(0,numpy.size(weights_history[3,:])))
    plot_weights_history3.set_ydata(weights_history[3,:])

    plot_lrate_history.set_xdata(numpy.arange(0,numpy.size(lrate_history)))
    plot_lrate_history.set_ydata(lrate_history)

    return line0, line1, selected_sample, plot_weights_history0, plot_weights_history1, plot_weights_history2, plot_weights_history3, plot_lrate_history

# def onclick(event):
#     global weights
#     weights = 4.0 * (numpy.random.random((2,2)) - 0.5)
#     update_data.epoch = 0   
# cid = fig.canvas.mpl_connect('button_press_event', onclick)


ani = animation.FuncAnimation(fig, updatefig, interval=20, init_func=init_draw, blit=True)
plt.show()