#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Reference:
# from website: http://jeremy.fix.free.fr/Softwares/neural_pca.html 

# Simulation of the Oja's covariance rule
import numpy 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to generate samples according to a gaussian distribution
# _________________________________________________________________

mu1 = 0.0
std1 = 1.0
mu2 = 0.0
std2 = 0.5
orientation = numpy.pi / 4.0#numpy.random.random() * numpy.pi
nb_samples = 200

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

print(numpy.shape(samples))

# Center the data
# _______________________________

samples[:,0] -= numpy.mean(samples[:,0])
samples[:,1] -=numpy.mean(samples[:,1])

# Functions to update the network
# _______________________________

# 2 inputs, 1 output 
weights = 2.0 * (numpy.random.random((2,1)) - 0.5)
weights_history = numpy.reshape(weights,(2,1))

initial_lrate = 0.4
lrate_history = numpy.array([initial_lrate])
presented_example = numpy.zeros((2,1))

def update_data():
    global weights, presented_example, weights_history, lrate_history

    # We take a random sample
    index_sample = numpy.random.randint(0,nb_samples)

    # prestened_example[0] is the first neuron, presented_example[1] is the second neuron
    presented_example[0] = samples[index_sample,0]
    presented_example[1] = samples[index_sample,1]

    # We compute the output from this input, this should be y
    output = samples[index_sample,0] * weights[0] + samples[index_sample,1]*weights[1]
    
    # Update the weights with the oja's rule
    weights[0] = weights[0] + update_data.lrate * ( samples[index_sample,0]*output - weights[0] * output * output)
    weights[1] = weights[1] + update_data.lrate * ( samples[index_sample,1]*output - weights[1] * output * output)    


    if(weights_history.shape[1] >= update_data.maxepoch):
        weights_history = numpy.hstack((weights_history[:,1:],weights))
        lrate_history = numpy.append(lrate_history[1:], [update_data.lrate])
    else:
        weights_history = numpy.hstack((weights_history,weights))
        lrate_history = numpy.append(lrate_history, [update_data.lrate])

    update_data.epoch = update_data.epoch + 1

    # Update the learning rate
    update_data.lrate = update_data.initial_lrate / (float(update_data.epoch)/nb_samples+1.0)

update_data.epoch = 1
update_data.maxepoch = 10 * nb_samples # 10 presentations of all the samples
update_data.initial_lrate = initial_lrate
update_data.lrate = update_data.initial_lrate

# Functions to deal with the display
# __________________________________

fig = plt.figure(facecolor='white')
ax_samples = fig.add_subplot(211)
plt.xlabel('X position')
plt.ylabel('Y position')

ax_weights = fig.add_subplot(212)
plt.xlabel('Weight value')
plt.ylabel('Epoch')


ax_samples.plot(samples[:,0], samples[:,1],'o', markersize=7, label='Samples')
line, = ax_samples.plot([0.0, weights[0].item()], [0.0, weights[1].item()], color='#00FF00', linewidth = 3, label='Principal component')
selected_sample, = ax_samples.plot([presented_example[0]], [presented_example[1]], 'or', markersize = 7, label='Current sample')
# Draw a circle of unitary radius for checking the norms of the weights
ax_samples.plot(numpy.cos(numpy.linspace(0,2*numpy.pi,100)), numpy.sin(numpy.linspace(0,2*numpy.pi,100)),'k--');

ax_samples.set_xlim(-max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))), max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))))
ax_samples.set_ylim(-max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))), max(max(numpy.abs(samples[:,0])),max(numpy.abs(samples[:,1]))))
ax_samples.set_aspect('equal')
ax_samples.legend(loc=(1.2,0.5))

plot_weights_history0, = ax_weights.plot([0.0], weights_history[0,:], 'b', label='1st weight')
plot_weights_history1, = ax_weights.plot([0.0], weights_history[1,:], 'g', label='2nd weight')
plot_lrate_history, = ax_weights.plot([0.0], lrate_history, 'r', label='Learning rate')
ax_weights.set_xlim(0, update_data.maxepoch)
ax_weights.set_ylim(-1.5, 1.5)
ax_weights.legend()

def init_draw():
    line.set_data([],[])
    selected_sample.set_data([],[])
    plot_weights_history0.set_data([],[])
    plot_weights_history1.set_data([],[])
    plot_lrate_history.set_data([],[])
    return line, selected_sample, plot_weights_history0, plot_weights_history1, plot_lrate_history

def updatefig(i):
    global weights, lrate_history
    update_data()

    # Update the data to display
    line.set_data([0.0, weights[0].item()], [0.0, weights[1].item()])

    selected_sample.set_data([presented_example[0]], [presented_example[1]])

    plot_weights_history0.set_xdata(numpy.arange(0,numpy.size(weights_history[0,:])))
    plot_weights_history0.set_ydata(weights_history[0,:])

    plot_weights_history1.set_xdata(numpy.arange(0,numpy.size(weights_history[1,:])))
    plot_weights_history1.set_ydata(weights_history[1,:])

    plot_lrate_history.set_xdata(numpy.arange(0,numpy.size(lrate_history)))
    plot_lrate_history.set_ydata(lrate_history)

    return line, selected_sample, plot_weights_history0, plot_weights_history1, plot_lrate_history

ani = animation.FuncAnimation(fig, updatefig, interval=20, init_func=init_draw, blit=True)
plt.show()