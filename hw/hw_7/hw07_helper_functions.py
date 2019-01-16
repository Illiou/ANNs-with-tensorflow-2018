# Download the dataset.

vocab_size = 20000
training, validation = tf.keras.datasets.imdb.load_data(
                        path='imdb.npz',
                        num_words=vocab_size,
                        skip_top=0,
                        maxlen=None,
                        seed=113,
                        start_char=1,
                        oov_char=2,
                        index_from=3
                    )

training_texts, training_labels = training
validation_texts, validation_labels = validation



# Cut off and fill all texts to a length of 300

def prepare_texts(texts, cutoff_length):    
    # cutoff all texts
    cutoff = [text[:cutoff_length] for text in texts]
    
    # fill short texts with zeros
    zeropad = [np.pad(text, (0,cutoff_length-len(text)), 'constant', constant_values=(0, 0)) for text in cutoff]
    
    return np.array(zeropad)


cutoff_length = 300

training_texts = prepare_texts(training_texts, cutoff_length)
validation_texts = prepare_texts(validation_texts, cutoff_length)


# Print a sample text.

word_to_id = tf.keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in training_texts[2] ))