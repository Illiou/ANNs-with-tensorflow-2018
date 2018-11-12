def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def create_training_tensors_cifar(directory):
    training_data = []
    training_labels = []
    for i in range (1,6):
        filename = directory + 'data_batch_{}'.format(i)
        dictionary = unpickle(filename)
        
        data = dictionary[b'data']
        data = np.reshape(data, newshape=(10000, 3, 32, 32))
        data = np.swapaxes(data, 1, 3)
        data = np.swapaxes(data, 1, 2)
        training_data.append(data)
        
        labels = dictionary[b'labels']
        training_labels.append(labels)
        
    training_data = np.array(training_data)
    training_data = np.reshape(training_data, newshape=(50000,32,32,3))
    training_labels = np.array(training_labels)
    training_labels = np.reshape(training_labels, newshape=(50000,))
    
    return training_data, training_labels
        
    
def create_validation_tensors_cifar(directory):

    filename = directory + 'test_batch'
    dictionary = unpickle(filename)

    data = dictionary[b'data']
    data = np.reshape(data, newshape=(10000, 3, 32, 32))
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 1, 2)
    
    labels = dictionary[b'labels']
    labels = np.array(labels)
    
    return data, labels
               

        
training_data, training_labels = create_training_tensors_cifar('./cifar-10-python/cifar-10-batches-py/')
validation_data, validation_labels = create_validation_tensors_cifar('./cifar-10-python/cifar-10-batches-py/')


label_to_word = {
    0: "Airplane",
    1: "Autombile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}
