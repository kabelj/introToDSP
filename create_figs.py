# This script creates the plots for AA6XA's Intro to DSP talk
#
# Change the frequencies and stuff at the top, or at the top of each section to 
# see how it changes the plots. Have fun, and experiment.
#

import numpy as np
import matplotlib.pyplot as plt

#----------------------------------
# Other variables
freq = 1.2*np.pi
num_bits = 3
levels = 2**num_bits
quant = levels/2 -0.1 #subtract 0.1 to make the quantization work correctly

# Time base
t = np.arange(0.0,10.0,0.01)
x = np.arange(0.0, 10.0, 0.2)
# CT
y = np.cos(freq*t)
z = np.floor(quant*y)
# DT
y_dt = np.cos(freq*x)
z_dt = np.floor(quant*y_dt)

#----------------------------------
# Continuous Analog
plt.figure(1)
plt.plot(t,y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Analog, CT, Signal')
#plt.legend(loc='upper right')

#----------------------------------
# Continuous Discrete
plt.figure(2)
plt.plot(t,z)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Digital, CT, Signal')

#----------------------------------
# Analog Discrete
plt.figure(3)
plt.stem(x,y_dt)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Discrete Time Signal')

plt.figure(4)
plt.plot(t,y,'r:')
plt.stem(x,y_dt)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Analog Discrete Time Signal')

#----------------------------------
# Digital Discrete
plt.figure(5)
plt.stem(x, z_dt)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Digitial Discrete Time Signal')

plt.figure(6)
plt.plot(t,z, 'r:')
plt.stem(x, z_dt)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Digitial, Discrete Time Signal')

#----------------------------------
# Nyquist and sampling
freq = 2*np.pi
sample_rate = 10 
t = np.arange(0.0, 10.0, 0.01)
x = np.arange(0.0, 10.0, 1.0/sample_rate)

y_ct = np.cos(freq*t)
y_dt = np.cos(freq*x)

plt.figure(7)
plt.subplot(2,2,1)
plt.plot(t,y_ct,'r', label='Original')
plt.stem(x,y_dt, label='Sampled')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Nyquist Sampling')


sample_rate = 2
x = np.arange(0.0, 10.0, 1.0/sample_rate)
y_dt = np.cos(freq*x)
plt.subplot(2,2,2)
plt.plot(t,y_ct,'r', label='Original')
plt.stem(x,y_dt, label='Sampled')
plt.xlabel('Time')
plt.ylabel('Amplitude')

sample_rate = 1.25
x = np.arange(0.0, 10.0, 1.0/sample_rate)
y_dt = np.cos(freq*x)
plt.subplot(2,2,3)
plt.plot(t,y_ct,'r', label='Original')
plt.stem(x,y_dt, label='Sampled')
plt.xlabel('Time')
plt.ylabel('Amplitude')

sample_rate = 1
x = np.arange(0.0, 10.0, 1.0/sample_rate)
y_dt = np.cos(freq*x)
plt.subplot(2,2,4)
plt.plot(t,y_ct,'r', label='Original')
plt.stem(x,y_dt, label='Sampled')
plt.xlabel('Time')
plt.ylabel('Amplitude')

#----------------------------------
plt.show()
