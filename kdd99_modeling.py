#inmport data analyse library
import pandas

#importing the dataset
dataset = pandas.read_csv('C:\IoT_security\Data_set\Dataset test\kddcup.data\kddcup.data.corrected')

#change Multi-class to binary-class
dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values
#optional
print("signal_types: {}".format(x))
print("classified: {}".format(y))

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])
onehotencoder_1 = OneHotEncoder(categorical_features = [1])
x = onehotencoder_1.fit_transform(x).toarray()
onehotencoder_2 = OneHotEncoder(categorical_features = [4])
x = onehotencoder_2.fit_transform(x).toarray()
onehotencoder_3 = OneHotEncoder(categorical_features = [70])
x = onehotencoder_3.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu', input_dim = 113))

#Adding a second hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

#Adding a third hidden layer
classifier.add(Dense(output_dim = 60, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
#optional
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("accuracy= ", cm)

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
#keras.utils.to_categorical(y, num_classes=None, dtype='float32')
from keras.utils import to_categorical
import numpy as np
conv = to_categorical(y_train)
y_train1 = np.argmax(conv,axis=1)
classifier.fit(x_train, y_train1, verbose=1, batch_size = 118, nb_epoch = 20)

#retrain and print training chart.
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.figure()

history = classifier.fit(x_train, y_train1, validation_split=0.25, epochs=20, batch_size=118, verbose=1)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#the performance of the classification model
print("the Accuracy is: "+ str((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])))
recall = cm[1,1]/(cm[0,1]+cm[1,1])
print("Recall is : "+ str(recall))
print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))
precision = cm[1,1]/(cm[1,0]+cm[1,1])
print("Precision is: "+ str(precision))
print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))
from math import log
print("Entropy is: "+ str(-precision*log(precision)))

#print model "classifier" graph
from keras.utils import plot_model
plot_model(classifier, to_file='model.png')

#print dataset chart
%matplotlib inline
import csv
from matplotlib import pyplot as plt
filename = 'C:\IoT_security\Data_set\Dataset test\kddcup.data\kddcup.data.corrected'
with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        signals = []
        classified = []
        for row in reader:
            signals.append(row[8])  
        print(signals)
        for column in reader:
            classified.append(column[8])  
        print(classified) 
        
        #Plot Data
        fig = plt.figure(dpi = 128, figsize = (10,6))
        plt.plot(signals, c = 'red') #Line 1
        plt.plot(classified, c = 'red')
        #Format Plot
        plt.title("Kddcup99 data visualize", fontsize = 24)
        plt.xlabel('signals',fontsize = 16)
        plt.ylabel("classified", fontsize = 16)
        plt.tick_params(axis = 'both', which = 'major' , labelsize = 16)
        plt.show()