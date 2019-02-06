#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
#import __future__
#import codecs
import sklearn.manifold as sk
import os
import multiprocessing
import logging
import nltk
import pandas as pd
import gensim.models.word2vec as w2v
#import seaborn as sb              
import numpy as np


# In[38]:

def sentence_to_wordlist(raw):
    clean=re.sub("[^a-zA-Z0-9]"," ",raw)
    words=clean.split()
    return words



def get_add_vec():
    

    nltk.download('punkt')
    nltk.download('stopwords')
    # In[39]:

    #os.system('del datasets/word_embedding.csv')
    df=pd.read_csv('./datasets/train_data_prep.csv')
    df.dropna()
    df=df.loc[:,]

    corpus_raw=u""
    for i in df['breaked_addr']:
        corpus_raw+=str(i);
        corpus_raw+=' ';
    print('corpus is now %i char long' %len(corpus_raw))
    corpus_raw=corpus_raw.replace('[',' ')
    corpus_raw=corpus_raw.replace(']',' ')
    corpus_raw=corpus_raw.replace(',',' ')
    corpus_raw=corpus_raw.replace('(',' ')
    corpus_raw=corpus_raw.replace(')',' ')
    corpus_raw=re.findall(r"[^\W\d_]+|\d+", corpus_raw)
    temp_corpus_raw=""
    for i in corpus_raw:
        temp_corpus_raw+=i+" "
    corpus_raw=temp_corpus_raw.strip()
    print(corpus_raw)
    input('PRESS ENTER....')

    tokens=corpus_raw.split(' ')
    print('tokens are now %i in size' %len(tokens))

    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences=tokenizer.tokenize(corpus_raw)

        #print(raw_sentences)

    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(sentence_to_wordlist(raw_sentence))

        #print(raw_sentences[10])
        #print(sentence_to_wordlist(raw_sentences[10]))

    token_count=sum([len(sentence) for sentence in sentences])  
    print("The address corpus %i tokens" %token_count)


        # In[40]:


    num_features=300
    min_word_count=3
    num_workers=multiprocessing.cpu_count()
    context_size=7
    downsampling=1e-1
    seed=1


        # In[41]:


    addr2vec=w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )


        # In[42]:


    addr2vec.build_vocab(sentences)
    print(len(addr2vec.wv.vocab))


        # In[43]:


    addr2vec.train(sentences,total_examples=len(sentences),epochs=num_workers)


        # In[44]:


    if not os.path.exists('addr_vocab'):
        os.makedirs('addr_vocab')
    addr2vec.save(os.path.join('addr_vocab','addr2vec.w2v'))


        # In[45]:


    tsne=sk.TSNE(n_components=2,random_state=0) 
    all_word_vectors_matrix=addr2vec.wv.syn0
    all_word_vectors_matrix_2d=tsne.fit_transform(all_word_vectors_matrix)


        # In[46]:


    points=pd.DataFrame(
        [
            (word,coords[0],coords[1])
            for word,coords in [
                (word,all_word_vectors_matrix_2d[addr2vec.wv.vocab[word].index])
                 for word in addr2vec.wv.vocab
        ]
        ],
        columns=['word','x','y']
    )


        # In[47]:


        #sb.set_context('poster')
        #points.plot.scatter('x','y',s=10)


        # In[48]:

    points.to_csv('./datasets/word_embedding.csv')
        #return points
    return

def get_add_vec_with_file(filename):
    

    nltk.download('punkt')
    nltk.download('stopwords')
    # In[39]:

    #os.system('del datasets/word_embedding.csv')
    df=pd.read_csv('./datasets/%s.csv'%filename)
    df.dropna()
    df=df.loc[:,]

    corpus_raw=u""
    for i in df['breaked_addr']:
        corpus_raw+=str(i);
        corpus_raw+=' ';
    print('corpus is now %i char long' %len(corpus_raw))
    corpus_raw=corpus_raw.replace('[',' ')
    corpus_raw=corpus_raw.replace(']',' ')
    corpus_raw=corpus_raw.replace(',',' ')
    corpus_raw=corpus_raw.replace('(',' ')
    corpus_raw=corpus_raw.replace(')',' ')
    corpus_raw=re.findall(r"[^\W\d_]+|\d+", corpus_raw)
    temp_corpus_raw=""
    for i in corpus_raw:
        temp_corpus_raw+=i+" "
    corpus_raw=temp_corpus_raw.strip()
    print(corpus_raw)
    input('PRESS ENTER...')

    tokens=corpus_raw.split(' ')
    print('tokens are now %i in size' %len(tokens))

    tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences=tokenizer.tokenize(corpus_raw)

        #print(raw_sentences)

    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(sentence_to_wordlist(raw_sentence))

        #print(raw_sentences[10])
        #print(sentence_to_wordlist(raw_sentences[10]))

    token_count=sum([len(sentence) for sentence in sentences])  
    print("The address corpus %i tokens" %token_count)


        # In[40]:


    num_features=300
    min_word_count=3
    num_workers=multiprocessing.cpu_count()
    context_size=7
    downsampling=1e-1
    seed=1


        # In[41]:


    addr2vec=w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )


        # In[42]:


    addr2vec.build_vocab(sentences)
    print(len(addr2vec.wv.vocab))


        # In[43]:


    addr2vec.train(sentences,total_examples=len(sentences),epochs=num_workers)


        # In[44]:


    if not os.path.exists('addr_vocab'):
        os.makedirs('addr_vocab')
    addr2vec.save(os.path.join('addr_vocab','addr2vec.w2v'))


        # In[45]:


    tsne=sk.TSNE(n_components=2,random_state=0) 
    all_word_vectors_matrix=addr2vec.wv.syn0
    all_word_vectors_matrix_2d=tsne.fit_transform(all_word_vectors_matrix)


        # In[46]:


    points=pd.DataFrame(
        [
            (word,coords[0],coords[1])
            for word,coords in [
                (word,all_word_vectors_matrix_2d[addr2vec.wv.vocab[word].index])
                 for word in addr2vec.wv.vocab
        ]
        ],
        columns=['word','x','y']
    )


        # In[47]:


        #sb.set_context('poster')
        #points.plot.scatter('x','y',s=10)


        # In[48]:
    if os.path.exists('./datasets/word_embedding.csv'):
        df_old=pd.read_csv('./datasets/word_embedding.csv')
        points=df_old.append(points)
        points=points.drop_duplicates()
        points.to_csv('./datasets/word_embedding.csv') 
    else:
        points.to_csv('./datasets/word_embedding.csv')
        #return points
    return

if __name__=='__main__':
    get_add_vec()
    print('[INFO] FIRST PART IS EXECUTED SUCCESSFULLY....')
    filename=input('enter file name:')
    get_add_vec_with_file(filename)
    

    print('PROGRAM EXECUTED SUCCESSFULLY')



# In[ ]:





# In[ ]:























































# CODED BY SPIDERN3MO