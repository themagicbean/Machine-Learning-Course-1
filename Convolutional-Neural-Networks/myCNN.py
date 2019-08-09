# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
L1 - overview
Need to have accurate classifications in data set
But data set no longer excel / csv, rather folder of images
1.  must separate images into training and test set folder,
2.  Need accurate classifications
Solutions:
    1.  Name files -- cat1 cat2 cat3 / cat1 cat2 cat3 etc. 
    2.  Better: included in Keras: one folder for cats, one folder for dogs
    
here using 10k total images, 5k cat / dog, 4k ea train / 1k ea test

data preprocessing MOSTLY no longer needed (feature scaling is)
no categorical data, so no encoding
train / test set already made
but need feature scaling b/c values of colors /position are arbitrary
(will take out of data preprocessing b/c rest not needeed)
    
"""

# ---- L2 - start building CNN - import modules

from keras.models import Sequential # initialize NN (as opposed to graph)
from keras.layers import Convolution2D  #2D b/c image
from keras.layers import MaxPooling2D # ditto
from keras.layers import Flatten #convert pooled feature maps to large feature vector
from keras.layers import Dense # add fully connected layers in classic ANN

# using TensorFlow backend

# ----L3 intialize CNN via Sequential package ... one line

classifier = Sequential()



# ----L4 adding layers, beginning w/ Convo layer ... another one line
# applies feature detector to input image, gives (x) product -> feature map
# inpute image (x) feature dtectro = feature map
# using many FDs to create many FMs = convolutional layer

classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
# args: 
#   np_filter: # of convo filters; nb_row, nb_col give rows and columns of FD
#      # frequently start w/ low # 32, 3x3, better for CPU, can go higher later
#       deprecations warning re first arg x (y1, ys))
#
#   border_mode = how to specify borders, default / usually use same   
#
#   input_shape = shape of input image -- need to force inputs to standard format
#       #s 3 = RGB ( 2 = B&W), next 2 #s are w/h in pixels
#           again, smaller format = quicker results
#       THEANO BACKEND IS DIFF FROM TENSOR FLOW
#       in TF, use w, h, d order for input_shape arg
#   activation = rectifier function (get rid of neg pixels), typ relu




# ---- L5 Pooling: reduce size of FMs via subtables to get pooled FM
# reduced FM layer called pooling layer.  Reduces nodes in next step

classifier.add(MaxPooling2D(pool_size = (2,2)))
# args:
#   pool_size = size of subtable, frequently 2x2




# --- here is where you would add extra layers (L10)
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# deleted input shape b/c input no longer preprocessed images
# rather, input now is the pooled feature maps from previous two steps
# may also want to up # of filters (first arg), not done here

classifier.add(MaxPooling2D(pool_size = (2,2)))




# ---- L6: Flattening: Make pooled FMS into single (huge) vector 
# high numbers preserved byP PFMs keeps spacial structure info
# if had just flattened image into vector, would not have image re:
# spacial connection b/t pixels (which was the goal)

classifier.add(Flatten())
# args: not needed



# ---- L7: create classic ANN using flattened vectors as input vectors
# aka the "full connection" step -- ading hidden / full connection + output

classifier.add(Dense(128, activation = 'relu'))
# args
#   order of args is depreciated (reverse) and output_dim renamed "units", first arg
#   output dim is # of nods in hidden layer
#       old rule of thumb, choose # b/t input # and output #
#       but here, ton of input nodes ... experiment to balance accuracy and power
#   activation fn = how/how much relevance each node passes on

# now adding output layer
classifier.add(Dense(1, activation = 'sigmoid'))
# args
#   see above note on deprecation
#   ouput dim to 1 b/c predicted probability of one class (dog or cat)
#   sigmoid act'n f'n reduces to binary probability
# also first arg is units, set to 1 here



# ---- L8: need to add grad descent, loss fn, perf. metric
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# deprectation warning, colocations halded auto by placer
# args
#   optimizer
#       adam function?

#   loss function "loss"
#       using binary b/c logistic regression and binary outcome
#       if more outcomes, cross-entropy (non-binary)

#   performance metric  "metrics" *NOTE: is a list *
#       accuracy = most common


# --- END OF BUILDING CNN





# ---- L9 fitting CNN to images
# here, can use Keras tools. But first! Image augmentation
# image augmentation -> helps to reduce overfitting
#   data augmentation: creates batches of images, applies random transforms
#   so increases diversity of images (more training material)
#   can use flow or flow_from_directory (here, latter fits dir structure)
# sample code on Keras doc will augment, fit, test ... does everything left
#   can use other transformations, etc.
#   renamed train_generator and validation_generatrion training_ and test_set

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255, # rescales pixels to bt 0 and 1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) # also have vert flip, n/ used

test_datagen = ImageDataGenerator(rescale=1./255) # same rescale

training_set = train_datagen.flow_from_directory(
        'dataset/training_set', # set wd for train set
        target_size=(64, 64), # match size of images expected by CNN
        batch_size=32, # size of batch including random samples
        class_mode='binary')
# gives output id'ing # of images & classes

test_set = test_datagen.flow_from_directory( # same mods
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
# ditto training set

classifier.fit_generator( # renamed model "classifier
        training_set, # renamed
        steps_per_epoch=8000, # renumbered to match # of images in trg set
        epochs=1, # changed to one just to follow tutorial, orign = 25
        validation_data=test_set, # renamed 
        validation_steps=2000) # formerly md_val_samples or sth like that
            # of test sets

# deprecation warning, ? use tf.cast instead?

# on old-ass computer took four hours per epoch .. bewarned
            
# compare training to test set accuracy
            
            
# ---- L10
# to make more accurate (w/o paramtere tuning), add more convolutional layers
# can also add more fully connected layer(s)
# repeat add convo layer + max pool function
# can also add extra feature detectors (e.g., 64) to extra convo layers

# higher target size would also help (more data)

