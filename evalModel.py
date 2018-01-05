from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

def word_for_id(integer, tokenizer):
    for word,index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],maxlen = max_length)
        yhat = model.predict([photo,sequence],verbose = 0)
        yhat = argmax(yhat)
        word = word_for_id(yhat,tokenizer)
        if word is None:
            break
        in_text += ' ' + word
    return in_text

def evaluate_model(model,description,photos,tokenizer,max_length):
    actual,predicted = list(),list()
    for key, desc_list in description.items():
        yhat = generate_desc(model, tokenizer, photos, max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    print('BLEU-1 %f' % corpus_bleu(actual,predicted,weights=(1.0,0,0,0)))
    print('BLEU-2 %f' % corpus_bleu(actual,predicted,weights=(0.5,0.5,0,0)))
    print('BLEU-3 %f' % corpus_bleu(actual,predicted,weights=(0.3,0.3,0,0)))
    print('BLEU-4 %f' % corpus_bleu(actual,predicted,weights=(0.25,0.25,0.25,0.25)))
