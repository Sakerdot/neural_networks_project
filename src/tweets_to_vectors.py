#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Author: Adrian Samatan

if __name__ == "__main__":
    import pickle
    import argparse
    import json
    import string
    import numpy as np
    import requests
    from tqdm import tqdm
    from nltk.tokenize import word_tokenize

    parser = argparse.ArgumentParser(
        description='Transforms the tweets into a GloVe word embedded vectors representation.')
    parser.add_argument('dataset_file', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('embeddings_file', type=str)
    parser.add_argument('vector_size', type=int)
    parser.add_argument('max_length', type=int)
    args = parser.parse_args()

    UNKNOWN_WORD = '<unk>'

    word_embeddings = {}
    with open(args.embeddings_file, 'r') as embeddings_file:
        for line in embeddings_file:
            splitted_line = line.split(' ')

            word_embedding = np.array(splitted_line[1:]).astype(np.float)
            word_embeddings[splitted_line[0]] = word_embedding

    tweets = []
    labels = []

    with open(args.dataset_file, 'r') as dataset_json_file:
        for tweet_str in dataset_json_file:
            tweet = json.loads(tweet_str)

            tweets.append(tweet['content'])
            labels.append(int(tweet['annotation']['label'][0]))

    def tokenize_tweet(tweet):
        return word_tokenize(tweet.translate(str.maketrans('', '', string.punctuation)))

    tokenized_tweets = [tokenize_tweet(tweet) for tweet in tweets]

    def obtain_definition(word):
        url = 'http://api.urbandictionary.com/v0/define?term=' + word
        resp = requests.get(url)
        res = resp.json()
        if 'list' in res and len(res['list']) != 0:
            first_def = res['list'][0]['definition']
            first_def_wo_punctuation = first_def.translate(
                str.maketrans('', '', string.punctuation))
            tokenized_definition = word_tokenize(first_def_wo_punctuation)

            definition_vector = np.zeros(args.vector_size)

            count = 0
            for i, word in enumerate(tokenized_definition):
                if word in word_embeddings:
                    count += 1
                    definition_vector += word_embeddings[word]
                
            if count == 0:
                return word_embeddings[UNKNOWN_WORD]

            definition_vector /= count

            return definition_vector
        else:
            return word_embeddings[UNKNOWN_WORD]

    output = []
    for i, words in tqdm(enumerate(tokenized_tweets), total=len(tokenized_tweets)):
        if len(words) == 0:
            continue
        else:
            tweet_vectors = np.zeros((args.max_length, args.vector_size), dtype=np.float)

            padding_start_pos = args.max_length - len(words)
            for j, word in enumerate(words):
                if word in word_embeddings:
                    tweet_vectors[j +
                                  padding_start_pos] = word_embeddings[word]
                else:
                    tweet_vectors[j +
                                  padding_start_pos] = obtain_definition(word)
            
            output.append((tweet_vectors, labels[i]))

    with open(args.output_file, 'wb') as output_file:
        pickle.dump(output, output_file)
