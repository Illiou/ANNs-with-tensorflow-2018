def tokenize_text(text):
    text_lower = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    text_tokenized = tokenizer.tokenize(text_lower)
    return text_tokenized

def create_dicts_from_tokenized_text(tokenized_text,vocabulary_size):
    words_and_count = Counter(tokenized_text).most_common(vocabulary_size - 1)
    print(words_and_count)
    word2id = {word: word_id for word_id, (word, _) in enumerate(words_and_count, 1)}
    word2id["_UNKNOWN_"] = 0
    id2word = dict(zip(word2id.values(), word2id.keys()))
    return word2id, id2word

def find_and_print_nearest_neighbors(target_words, number_of_nearest_neighbors):
    embedding_values = sess.run(embeddings)
    normed_embeddings = embedding_values / np.sqrt(np.sum(embedding_values**2, axis=1, keepdims=True))
    for word in target_words:
        word_id = word2id[word]
        word_embedding = normed_embeddings[word_id, :]
        cosine_similarities = np.matmul(normed_embeddings, word_embedding )
        n_nearest_neighbors = np.argsort(-cosine_similarities)[:number_of_nearest_neighbors]
        print("Nearest to " + word + ": " + ", ".join([id2word[nearest] for nearest in n_nearest_neighbors]))