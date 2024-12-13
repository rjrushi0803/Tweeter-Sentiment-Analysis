"""This file contains important functions"""
import pandas as pd 
import numpy as np 
import spacy
import pickle

## preprocessing tweets column
nlp = spacy.load('en_core_web_sm')

## function to preprocess
def lemmatization(text):
    doc = nlp(text)
    lemm_ls = [word.lemma_ for word in doc]
    return ' '.join(lemm_ls)

## remove stopwords
def rm_stop(text):
    doc = nlp(text)
    ns = [word.text for word in doc if not word.is_stop and not word.is_punct]
    return " ".join(ns)

## loading trained models
with open('game_encoder.pkl','rb') as f:
    game_encode = pickle.load(f)

with open('tfidf_vect_model.pkl','rb') as f:
    tfidf_vect = pickle.load(f)

with open('random_forest_model.pkl','rb') as f:
    rf_model = pickle.load(f)

## function to predict
def make_predictions(x):
    """function takes dataframe made out of user inputs"""
    game_encoded = game_encode.transform(x[['Game_name']])
    game_df = pd.DataFrame(game_encoded, columns=game_encode.get_feature_names_out())
    
    ## applying lemmatization
    x['lemma'] = x['tweets'].apply(lemmatization)

    ## removing stop words
    x['final'] = x.lemma.apply(rm_stop)

    # x_df = pd.concat([x['final'],game_df],axis=1)
    tfidf_matrix = tfidf_vect.transform(x['final'])
    vect_df = pd.DataFrame(tfidf_matrix.toarray(),columns= tfidf_vect.get_feature_names_out())

    ## concatenate 
    x_final = pd.concat([game_df,vect_df],axis=1)

    ## make predictions
    return rf_model.predict(x_final)
