# Script for creating FFT and filtering plots
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#----------------------------------
# FFT
Fs = 1000
dt = 1/Fs
t = np.arange(0.0,3.0,dt)
N = t.size
dF = Fs/N

freq1 = 20
freq2 = 150
freq3 = 400

x = np.cos(2*np.pi*freq1*t)
y = np.cos(2*np.pi*freq2*t)
z = np.sin(2*np.pi*freq3*t)
sig = x+y+z

S = np.fft.fft(sig,n=512)
f = np.arange(-Fs/2,Fs/2,Fs/512)

plt.figure(10)
plt.plot(t,sig)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid()
plt.title('Three Sine Waves')

plt.figure(11)
plt.plot(f,np.abs(np.fft.fftshift(S))/N)
plt.xlabel('frequency, Hz')
plt.ylabel('amplitude')
plt.grid(which='both')
plt.title('Sine Wave FFT')

#----------------------------------
#square wave
Fs = 1000
t = np.linspace(0,1,Fs)
N = t.size
freq = 25

x = sp.signal.square(2*np.pi*freq*t)
X = np.fft.fft(x,n=512)
f = np.arange(-Fs/2,Fs/2,Fs/512)

plt.figure(12)
plt.plot(t,x)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid()
plt.title('Square Wave')

plt.figure(13)
plt.plot(f,np.abs(np.fft.fftshift(X))/N)
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.grid()
plt.title('Square Wave FFT')

#==================================
#----------------------------------
# Nice LPF
Fs = 1000
cutoff = 200
order = 2

b,a = sp.signal.butter(order, cutoff, analog=False, fs=Fs)
w,h = sp.signal.freqz(b, a, fs=Fs)

plt.figure(14)
plt.semilogx(w, 20*np.log10(abs(h)))
plt.axvline(cutoff, color='red')
plt.xlabel('frequency, Hz')
plt.ylabel('amplitude, dB')
plt.grid()
plt.title('2nd Order Butterworth Filter')

H = np.fft.ifft(h,n =1024)
plt.figure(15)
plt.plot(abs(np.fft.fftshift(H)))
plt.xlabel('sample number')
plt.ylabel('amplitude')
plt.grid()
plt.title('Butterworth Time Domain')

#----------------------------------
# Brick Wall LPF
Fs = 1000
cutoff = 200

t = np.linspace(-1,1,Fs)
N = t.size

#x = np.concatenate([np.zeros(300,dtype=int), np.ones(200,dtype=int), np.ones(200,dtype=int), np.zeros(300,dtype=int)])
x = np.concatenate([np.ones(200,dtype=int), np.zeros(300,dtype=int)])
X = np.fft.fft(x,n=2048)

plt.figure(16)
plt.semilogx(np.linspace(0,499, num=500),x)
plt.grid()
plt.title('Ideal (Brick Wall) LPF')

plt.figure(17)
plt.plot(np.abs(np.fft.fftshift(X))/N)
plt.xlabel('sample number')
plt.ylabel('amplitude')
plt.grid()
plt.title('Ideal (Brick Wall) Time Domain')

#Reverse
xx = np.fft.ifft(X,n=512)
plt.figure(30)
plt.semilogx(xx)

#----------------------------------
# Chebyshev Filter (type 1)
Fs = 1000
cutoff = 200
order = 2
ripple = 2 #dB

b,a = sp.signal.cheby1(order, ripple, cutoff, btype='low', analog=False, fs=Fs)
w,h = sp.signal.freqz(b, a, fs=Fs)

plt.figure(18)
plt.semilogx(w, 20*np.log10(abs(h)))
plt.axvline(cutoff, color='red')
plt.xlabel('frequency, Hz')
plt.ylabel('amplitude, dB')
plt.grid()
plt.title('Chebyshev 2nd Order LPF')

H = np.fft.ifft(h,n =1024)
plt.figure(19)
plt.plot(abs(np.fft.fftshift(H)))
plt.xlabel('sample number')
plt.ylabel('amplitude')
plt.grid()
plt.title('Chebyshev Time Domain')

#----------------------------------
# Signal Chain Example
Fs = 1000
dt = 1/Fs
t = np.arange(0.0,1.0,dt)
N = t.size
dF = Fs/N
cutoff = 200
order = 2
nFFT = 1024
f = np.arange(-Fs/2,Fs/2,Fs/nFFT)

freq1 = 20
freq2 = 150
freq3 = 400

x = np.cos(2*np.pi*freq1*t)
y = np.cos(2*np.pi*freq2*t)
z = np.sin(2*np.pi*freq3*t)
sig = x+y+z

S = np.fft.fft(sig,n=nFFT)
b,a = sp.signal.butter(order, cutoff, analog=False, fs=Fs)
w,h = sp.signal.freqz(b, a, fs=Fs)
filt_out = sp.signal.lfilter(b,a,sig)
s = np.fft.fft(filt_out,n=nFFT)

plt.figure(20)
plt.subplot(2,3,1)
plt.plot(t,sig)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid()
plt.title('Original Signal')

plt.subplot(2,3,4)
plt.plot(f,np.abs(np.fft.fftshift(S)/N))
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.grid()
plt.title('Original FFT')

plt.subplot(2,3,5)
plt.semilogx(w, 20*np.log10(abs(h)))
plt.axvline(cutoff, color='red')
plt.xlabel('frequency, Hz')
plt.ylabel('amplitude, dB')
plt.grid()
plt.title('Low Pass Filter')

plt.subplot(2,3,2)
H = np.fft.ifft(h,n =nFFT)
plt.plot(abs(np.fft.fftshift(H)))
plt.xlabel('sample number')
plt.ylabel('amplitude')
plt.grid()
plt.title('Butterworth Time Domain')

plt.subplot(2,3,3)
plt.plot(t,filt_out)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid()
plt.title('Filtered Signal')

plt.subplot(2,3,6)
plt.plot(f,np.abs(np.fft.fftshift(s))/N)
plt.xlabel('frequency, Hz')
plt.ylabel('amplitude')
plt.grid()
plt.title('Filtered FFT')

#----------------------------------
# Square Wave filtering
Fs = 1000
dt = 1/Fs
t = np.linspace(0,1,Fs)
N = t.size
dF = Fs/N
cutoff = 100
order = 4
nFFT = 1024
f = np.arange(-Fs/2,Fs/2,Fs/nFFT)
freq = 25

sig = sp.signal.square(2*np.pi*freq*t)

S = np.fft.fft(sig,n=nFFT)
b,a = sp.signal.butter(order, cutoff, analog=False, fs=Fs)
w,h = sp.signal.freqz(b, a, fs=Fs)
filt_out = sp.signal.lfilter(b,a,sig)
s = np.fft.fft(filt_out,n=nFFT)

plt.figure(21)
plt.subplot(2,3,1)
plt.plot(t,sig)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid()
plt.title('Original Signal')

plt.subplot(2,3,4)
plt.plot(f,np.abs(np.fft.fftshift(S)/N))
plt.xlabel('frequency')
plt.ylabel('amplitude')
plt.grid()
plt.title('Original FFT')

plt.subplot(2,3,5)
plt.semilogx(w, 20*np.log10(abs(h)))
plt.axvline(cutoff, color='red')
plt.xlabel('frequency, Hz')
plt.ylabel('amplitude, dB')
plt.grid()
plt.title('Low Pass Filter')

plt.subplot(2,3,2)
H = np.fft.ifft(h,n =nFFT)
plt.plot(abs(np.fft.fftshift(H)))
plt.xlabel('sample number')
plt.ylabel('amplitude')
plt.grid()
plt.title('Butterworth Time Domain')

plt.subplot(2,3,3)
plt.plot(t,filt_out)
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.grid()
plt.title('Filtered Signal')

plt.subplot(2,3,6)
plt.plot(f,np.abs(np.fft.fftshift(s))/N)
plt.xlabel('frequency, Hz')
plt.ylabel('amplitude')
plt.grid()
plt.title('Filtered FFT')

#----------------------------------
plt.show()
