import scipy.io

def load_svhn(path):
    
    matfile = scipy.io.loadmat(path)
    images = matfile['X']
    images = np.expand_dims(images,0)
    images = np.swapaxes(images, 0,4)
    images = np.squeeze(images)
    labels = matfile['y']
    labels = np.squeeze(labels)
    print(len(labels))
    
    validation_images = images[:10000,:,:,:]
    training_images = images[10000:,:,:,:]
    validation_labels = labels[:10000]
    training_labels = labels[10000:]
    
    return training_images, training_labels, validation_images, validation_labels


training_images, training_labels, validation_images, validation_labels = load_svhn('./SVHN/train_32x32.mat')