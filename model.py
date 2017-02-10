import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import batch_generator, input_shape

np.random.seed(0)


######################################
#
# Data
#
######################################

data_df = pd.read_csv('data/driving_log.csv')

X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


######################################
#
# Model
#
######################################

# Modified NVIDIA model
keep_prob = 0.5

model = Sequential()
model.add(Lambda(lambda x:x/127.5-1.0, input_shape=input_shape))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2,2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2,2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2,2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Dropout(keep_prob))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense( 50, activation='elu'))
model.add(Dense( 10, activation='elu'))
model.add(Dense(  1))
model.summary()

# save the model
model_json = model.to_json()
with open('model.json', 'w') as f:
    f.write(model_json)


######################################
#
# Training
#
######################################

batch_size = 40
samples_per_epoch = 20000
nb_epoch = 5

checkpoint = ModelCheckpoint("model-{epoch:03d}.h5",
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_images=True)

model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))

model.fit_generator(batch_generator(X_train, y_train, batch_size, True),
                    samples_per_epoch, 
                    nb_epoch,
                    max_q_size=1,
                    validation_data=batch_generator(X_valid, y_valid, batch_size, True),
                    nb_val_samples=len(X_valid),
                    callbacks=[checkpoint, tensorboard],
                    verbose=1)

