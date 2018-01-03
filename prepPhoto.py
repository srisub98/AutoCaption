#load each photo, prep for VGG, and collect predicted features from the VGG model
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

#extract features from each photo in the directory
def extract_features(directory):
    #load the model
    model = VGG16()
    #re-structure the model
<<<<<<< HEAD
    model.layers.pop() #remove the last layer which does the classification
=======
    model.layers.pop()
>>>>>>> d0677849d6c1921f78883aba9a39f99956d57daa
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    #summarize
    print(model.summary())
    #extract features from each photo
    features = dict()
    for name in listdir(directory):
        #load an image from file
        filename = directory + '/' + name
<<<<<<< HEAD
        try:
            image = load_img(filename, target_size=(224,224))
        except IOError:
            continue
=======
        image = load_img(filename, target_size=(224,224))
>>>>>>> d0677849d6c1921f78883aba9a39f99956d57daa
        #convert image pixels to a numpy array
        image = img_to_array(image)
        #reshape data for the model
        image = image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
        #prep for VGG model
        image = preprocess_input(image)
        #get extract_features
        feature = model.predict(image, verbose = 0)
        #get image id
        image_id = name.split('.')[0]
        #store feature
        features[image_id] = feature
        print('>%s' % name)
    return features

<<<<<<< HEAD
directory = '/Users/Sri/Desktop/Projects/TensorFlow/AutoCaption/Flicker8k_Dataset'
=======
directory = 'Flicker8k_Dataset'
>>>>>>> d0677849d6c1921f78883aba9a39f99956d57daa
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
#save to file
dump(features, open('features.pkl','wb'))
