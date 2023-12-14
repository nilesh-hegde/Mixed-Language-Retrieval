#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 23:22:35 2023
@author: Nilesh Hegde
CS 6200 - Information Retrieval
Project - Mixed-language retrieval

This file contains all the helper functions needed to implement the
project. All the necessary modules needed to run this file are stored in the
requirements.txt file and can be installed using it.
"""


import fasttext
from typing import Callable
import translators as ts
from nltk.tokenize import word_tokenize
from sacremoses import MosesDetokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
import math
from sortedcontainers import SortedList
from pyserini.analysis import Analyzer, get_lucene_analyzer
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import nltk

nltk.download('punkt')

FASTEXT_LANG_MODEL = fasttext.load_model("lid.176.bin")
LANGUAGES = {'en':'english','fr':'french','es':'spanish'}
BERT_MODEL = SentenceTransformer('bert-base-nli-mean-tokens')

def detect_language(word:str) -> str | None:
    '''
    This function returns the predicts the most probable language origin of a 
    given text using language detection model trained on fasttext
    (uncompressed version)

    Parameters
    ----------
    word : str
        word whose language needs to be detected

    Returns
    -------
    str : 
        most probable language predicted by the fasttext model
    NONE :
        if languages predicted not among given languages
        

    '''
    result = FASTEXT_LANG_MODEL.predict(word,5)
    for language in result[0]:
        if language[-2:] in LANGUAGES:
            return language[-2:]
    return None

def translate(translator: str, word: str,source: str,target: str) -> str:
    '''
    This function translates a word given the source and the target lannguage.
    The translation is dependent on the translator.

    Parameters
    ----------
    translator : Callable
        the translator used to translate
    word : str
        word that needs to be translated
    source : str
        source language of the given word
    target : str
        target language to which word needs to be translated to

    Returns
    -------
    str
        translated word

    '''
    translated_word = ts.translate_text(word,translator,source,target)
    if '(' in translated_word and ')' in translated_word:
        start = translated_word.index('(')
        end = translated_word.index(')')
        translated_word = translated_word[start+1:end]
    return translated_word

def pretty_print(document_text:str,query_words:list[str]) -> str:
    '''
    This function is used to find query words in text and highlight them
    and return highlighted text to make it easier to find information being
    searched using the query

    Parameters
    ----------
    document_text : str
        the copntents of the given document
    query_words : str
        words used in the query

    Returns
    -------
    str
        highlighted document text

    '''
    location = dict()
    document_tokens = word_tokenize(document_text)
    for word in query_words:
        try:
            idx = document_tokens.index(word)
            location[word] = idx
        except:
            pass
    display_words_list = list()
    for word in location:
        word_list = list()
        for i in range(-3,4):
            if i == 0:
                word_list.append('<b>' + word + '</b>')
                continue
            try:
                word_list.append(document_tokens[location[word]+i])
            except:
                pass
        display_words_list.append(word_list)
    display_str = ''
    detokenizer = MosesDetokenizer(lang='en')
    for word_list in display_words_list:
        display_str += ' ... ' + detokenizer.detokenize(word_list)
    display_str += ' ... '
    return display_str

def translate_query(translator: str, query: str) -> dict[str,str]:
    '''
    This function translates a mixed language query into queries in the
    English, Spanish and French language

    Parameters
    ----------
    translator : Callable
        the translator used to translate
    query : str
        mixed language query that needs to be translated

    Returns
    -------
    dict[str,str]
        queries in english, spanish and french languages

    '''
    words = query.split(' ')
    words_en = list()
    words_es = list()
    words_fr = list()
    for word in words:
        word_origin = detect_language(word)
        if word_origin == 'en':
            word_en = word
            if translator == 'baidu':
                word_fr = translate(translator, word, word_origin, 'fra')
            else:
                word_fr = translate(translator, word, word_origin, 'fr')
            if translator =='baidu':
                word_es = translate(translator, word, word_origin, 'spa')
            else:
                word_es = translate(translator, word, word_origin, 'es')
        elif word_origin == 'es':
            if translator == 'baidu':
                word_en = translate(translator, word, 'spa', 'en')
            else:
                word_en = translate(translator, word, word_origin, 'en')
            word_es = word
            if translator == 'baidu':
                word_fr = translate(translator, word, 'spa', 'fra')
            else:
                word_fr = translate(translator, word, word_origin, 'fr')
        elif word_origin == 'fr':
            if translator == 'baidu':
                word_en = translate(translator, word, 'fra', 'en')
            else:
                word_en = translate(translator, word, word_origin, 'en')
            if translator == 'baidu':
                word_es = translate(translator, word, 'fra', 'spa')
            else:
                word_es = translate(translator, word, word_origin, 'es')
            word_fr = word
        
        words_en.append(word_en)
        words_es.append(word_es)
        words_fr.append(word_fr)
    return {'english':' '.join(words_en),'spanish':' '.join(words_es),'french':' '.join(words_fr)}
        

def document_similarity(documents: list[str]) -> pd.DataFrame:
    '''
    This function uses the Multilingual Universal Sentence Encoder to find 
    similarity between top documents retrieved by the queries in different
    languages.

    Parameters
    ----------
    documents : list[str]
        documents retrieved by translation of queries in different languages.

    Returns
    -------
    similarity_df : pd.DataFrame
        similarity scores matrix between all documents

    '''
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    embeddings = embed(documents)
    similarities = cosine_similarity(embeddings)
    return pd.DataFrame(similarities,index=documents,columns=documents).style.background_gradient(axis=None)

def get_doc_id_set(token: str,reader: IndexReader,searcher: LuceneSearcher) -> set:
    '''
    This function returns the set of all ids for all documents where given
    token appears

    Parameters
    ----------
    token : str
        string that holds the token for which document ids need to be found
    reader : IndexReader
        IndexReader object
    searcher : LuceneSearcher
        LuceneSearcher object

    Returns
    -------
    set
        set of all document ids token exists in

    '''
    doc_id_set = set()
    posting_list = reader.get_postings_list(token, analyzer=None)
    for posting in posting_list:
        doc_id_set.add(searcher.doc(posting.docid).docid())
    return doc_id_set

def get_cf(token: str,reader: IndexReader) -> int:
    '''
    This function returns the collection frequency given the token

    Parameters
    ----------
    token : str
        string that holds the token for which document ids need to be found
    reader : IndexReader
        IndexReader object

    Returns
    -------
    int
        collection frequency of token

    '''
    df, cf = reader.get_term_counts(token, analyzer=None)
    return cf

def get_doc_vector(doc_id: str,reader: IndexReader) -> dict[str,int]:
    '''
    This function returns the document vector containing the frequencies of each
    word in the document associated with document id.

    Parameters
    ----------
    doc_id : str
        string that hold the document id for which vector needs to found
    reader : IndexReader
        IndexReader object

    Returns
    -------
    dict[str,int]
        dictionary of frquencies of each word in document

    '''
    doc_vector = reader.get_document_vector(doc_id)
    return doc_vector

def query_likelihood_model(query: str, reader: IndexReader, searcher: LuceneSearcher, k: int, language: str = 'en') -> SortedList:
    '''
    This function return the k documents with the highest query likelihoods 
    with Dirichlet smoothing

    Parameters
    ----------
    query : str
        string that holds the query for which top-k documents need to be found
    reader : IndexReader
        IndexReader object
    k : int
        interger specifying how many documents need to be returned
    language : str
        string holding the language of the query

    Returns
    -------
    SortedList
        sorted list of tuples holding the log-likelihood and the document as a 
        tuple

    '''
    mu = 1000
    analyzer = Analyzer(get_lucene_analyzer(language))
    tokens = analyzer.analyze(query)
    corpus_length = reader.stats()['total_terms']
    doc_ids = set()
    for token in tokens:
        doc_ids.update(get_doc_id_set(token,reader,searcher))
    hits = SortedList()
    for doc_id in doc_ids:
        doc_vector = get_doc_vector(doc_id,reader)
        doc_length = sum(doc_vector.values())
        log_likelihood = 0
        for token in tokens:
            if token in doc_vector:
                tf = doc_vector[token]
            else:
                tf = 0
            log_likelihood += math.log((tf + mu*(get_cf(token,reader)/corpus_length))/(doc_length + mu))
        hits.add((log_likelihood,doc_id,language))
        if len(hits) > k:
            hits.pop(0)
    return hits[::-1]

def okapi_bm25(query: str, reader: IndexReader, searcher: LuceneSearcher, k: int, language: str = 'en') -> SortedList:
    '''
    This function implements the Okapi BM25 language model and its matching
    score to retrieve top k documents based on this score.

    Parameters
    ----------
    query : str
        string that holds the query for which top-k documents need to be found
    reader : IndexReader
        IndexReader object
    k : int
        interger specifying how many documents need to be returned
    language : str
        string holding the language of the query

    Returns
    -------
    SortedList
        sorted list of tuples holding the log-likelihood and the document as a 
        tuple

    '''
    k1 = 1.2
    b = 0.75
    k2 = 0
    analyzer = Analyzer(get_lucene_analyzer(language))
    tokens = analyzer.analyze(query)
    corpus_length = reader.stats()['total_terms']
    D = reader.stats()['documents']
    avdl = corpus_length/D
    doc_ids = set()
    documents_per_token = dict()
    for token in tokens:
        doc_id_set = get_doc_id_set(token,reader,searcher)
        documents_per_token[token] = len(doc_id_set)
        doc_ids.update(doc_id_set)
    hits = SortedList()
    for doc_id in doc_ids:
        doc_vector = get_doc_vector(doc_id,reader)
        doc_length = sum(doc_vector.values())
        score = 0
        for token in tokens:
            if token in doc_vector:
                tf = doc_vector[token]
            else:
                tf = 0
            score += math.log((D+0.5)/(documents_per_token[token]+0.5))*((tf+tf*k1)/(tf+k1*((1-b)+b*(doc_length/avdl))))
        hits.add((score,doc_id,language))
        if len(hits) > k:
            hits.pop(0)
    return hits[::-1]

def okapi_tf_idf(query: str, reader: IndexReader, searcher: LuceneSearcher, k: int, language: str = 'en') -> SortedList:
    '''
    This function implements the TF-IDF vector space model with Okapi TF as the
    scoring function to retrieve top k documents based on the scores.

    Parameters
    ----------
    query : str
        string that holds the query for which top-k documents need to be found
    reader : IndexReader
        IndexReader object
    k : int
        interger specifying how many documents need to be returned
    language : str
        string holding the language of the query

    Returns
    -------
    SortedList
        sorted list of tuples holding the log-likelihood and the document as a 
        tuple

    '''
    analyzer = Analyzer(get_lucene_analyzer(language))
    tokens = analyzer.analyze(query)
    corpus_length = reader.stats()['total_terms']
    D = reader.stats()['documents']
    avdl = corpus_length/D
    doc_ids = set()
    documents_per_token = dict()
    for token in tokens:
        doc_id_set = get_doc_id_set(token,reader,searcher)
        documents_per_token[token] = len(doc_id_set)
        doc_ids.update(doc_id_set)
    hits = SortedList()
    for doc_id in doc_ids:
        doc_vector = get_doc_vector(doc_id,reader)
        doc_length = sum(doc_vector.values())
        score = 0
        for token in tokens:
            if token in doc_vector:
                tf = doc_vector[token]
            else:
                tf = 0
            score += (tf/(tf+0.5+1.5*(doc_length/avdl)))*math.log(D/documents_per_token[token])
        hits.add((score,doc_id,language))
        if len(hits) > k:
            hits.pop(0)
    return hits[::-1]