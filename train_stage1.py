#Create base voices for CNN and trane it
#Gubin M.
import matplotlib
matplotlib.use('Agg')

#from __future__ import print_function

import os
import numpy as np
from matplotlib import pyplot, mlab
import scipy.io.wavfile
from collections import defaultdict
from scipy.fftpack import fft
import random

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.optimizers import RMSprop
import h5py
import gc
import sys

#список wav файлов
list_wav_files = ['./WAVFile2/voiceA.wav',
    './WAVFile2/noiseMan1.wav',
    './WAVFile2/noiseMan2.wav',
    './WAVFile2/noiseWoman1.wav',
    './WAVFile2/noiseWoman2.wav',
    './WAVFile2/noiseMan2_Woman1.wav',
    './WAVFile2/noiseMan1_Woman2.wav',
    './WAVFile2/noiseMan2_Woman2.wav',
    './WAVFile2/voiceA_noiseMan1.wav',
    './WAVFile2/voiceA_noiseMan2.wav',
    './WAVFile2/voiceA_noiseWoman1.wav',
    './WAVFile2/voiceA_noiseWoman2.wav',
    './WAVFile2/noiseWoman1_Woman2.wav',
    './WAVFile2/noiseMan1_Man2.wav']

##list_y = [0, 1, 1, 1, 1] #, 0, 0, 0, 0, 1, 1]
list_y = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]   #for voiceA
#list_y = [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0]   #for Man1
#list_y = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]   #for Man2
#list_y = [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1]   #for Woman1
#list_y = [1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]   #for Woman2

#количество файлов wav
count_file=len(list_wav_files)
print('count files = ', count_file)


#function show spectrogram
SAMPLE_RATE = 8000 # Hz
WINDOW_SIZE = 1024 # размер окна, в котором делается fft
WINDOW_STEP = 32# шаг окна default=128
WINDOW_OVERLAP = 16 #default=64

#keras model param
batch_size = 30
num_classes = 2
epochs = 15
img_rows, img_cols = 513, 219

#start programm
model_path = 'new4_model_3levels-4.h5'
model=load_model(model_path)
model.summary()

#load base
file_base = 'validation_new4_a_280.npz'
f = np.load(file_base)
x_test=f['x_test']
y_test=f['y_test']

#
file_base ='base_words_A.npz'
f = np.load(file_base)
word_start = f['word_start']
word_end = f['word_end']
word_len = f['word_len']

#
#размер фрагмента в амплитудах для спектрограммы
maxRazmer = 8000
#максимальная длина wav файла
maxLenWavFile = 70000000


#смещение в цикле задается через командную строку
cicle_ml = int(sys.argv[1])

#число записей в обучающей выборке из одного файла wav
samples=80
bais_samples = 0
samples_start = samples*cicle_ml+bais_samples*samples

#всего записей в базе для обучения
full_samples = samples*count_file

#храним wav_data
wave_data = np.zeros(count_file*maxLenWavFile)
wave_data.shape=(count_file, maxLenWavFile)
print ('shape = ',wave_data.shape)

for i in range(count_file):
    #get wav_data and get params
    sample_rate, wave_data_hlp = scipy.io.wavfile.read(list_wav_files[i])
    print ('\n file wav ',i , "  = ",list_wav_files[i])
    print ("sample_rate = ",sample_rate)
    print("dtype = ", wave_data_hlp.dtype)
    print("shape = ", wave_data_hlp.shape[0])
    print("len = ", len(wave_data_hlp))
    #Get duration of sound file
    signalDuration =  wave_data_hlp.shape[0] / sample_rate
    print("signalDuration = ", signalDuration)
    wave_data[i] = wave_data_hlp[0:maxLenWavFile]


#find count full samples
count_num_spectogramm = -1

for j in range(samples_start,samples_start+samples,1):

    count_cicle = int((word_len[j] - maxRazmer)/2000)
    if (count_cicle<1): count_cicle=1

    for j_count_cicle in range(count_cicle):

        for i in range(count_file):
           count_num_spectogramm = count_num_spectogramm + 1



full_samples = count_num_spectogramm+1
print("\n full_samples = ", full_samples)
#quit()

#создаем массив состоящий из 0
array = np.zeros(maxRazmer);

x_train = np.zeros(full_samples*img_rows*img_cols)
x_train.shape=(full_samples, img_rows, img_cols)
y_train = full_samples*[0]
count_num_spectogramm = -1

for j in range(samples_start,samples_start+samples,1):

    count_cicle = int((word_len[j] - maxRazmer)/2000)
    if (count_cicle<1): count_cicle=1

    for j_count_cicle in range(count_cicle):

        number = word_start[j] +j_count_cicle*2000

        for i in range(count_file):
            #get from wav
            count_num_spectogramm=count_num_spectogramm+1
            array = wave_data[i, number:number+maxRazmer]
            x_hlp = pyplot.specgram(array,NFFT=WINDOW_SIZE, noverlap=WINDOW_SIZE - WINDOW_STEP, Fs=SAMPLE_RATE) #scale= 'linear'
            x_train[count_num_spectogramm] =  x_hlp[0]
            y_train[count_num_spectogramm] = list_y[i]
            print(count_num_spectogramm," ", list_wav_files[i]," ", y_train[count_num_spectogramm])
            #show_all(array)


del wave_data
del array
gc.collect()

print('shape baze')
print("x_train.shape = ",x_train[0].shape)
print('len train records x = ', len(x_train))
print('len train records y = ', len(y_train))
print('len test records x = ', len(x_test))
print('len test records y = ', len(y_test))

x_train /= 255
x_test /= 255

x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)######
y_test = keras.utils.to_categorical(y_test, num_classes)######

model.summary()

print('cicle_ml = ', cicle_ml)

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

print('cicle_ml = ', cicle_ml, ' ending')
model.save(model_path)
print('Saved trained model at %s ' % model_path)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#write to log file
file_log = open(model_path+".txt", 'a')
file_log.write("\n cicle_ml = "+str(cicle_ml))
file_log.write("  Test loss: "+ str(score[0]))
file_log.write("  Test accuracy: "+ str(score[1]))
file_log.close()

del x_test
del y_test
del x_train
del y_train

gc.collect()

print("end programm")

