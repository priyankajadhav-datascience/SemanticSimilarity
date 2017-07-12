
# coding: utf-8

# In[58]:

from nltk.stem import *
from nltk.corpus import sentiwordnet as swn
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re, math
from collections import Counter
import psycopg2
from django.db import transaction, DatabaseError
import numpy as np
from sklearn.cluster import KMeans
import time
import os
from flask_restplus import Api, Resource, fields, reqparse  
from flask_cors import CORS, cross_origin


# In[59]:

def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    fltr_sentence = []
    tokens = word_tokenize(text)
    for txt in tokens:
        if txt not in stop_words:
            fltr_sentence.append(txt)
    new_sentence=" ".join(fltr_sentence)
    return(new_sentence)
def stemma(text):
    port = PorterStemmer()
    stemma=[port.stem(i) for i in text.split()]
    wnl = WordNetLemmatizer()
    lemma_tex=" ".join([wnl.lemmatize(i) for i in stemma]) 
    return lemma_tex
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    return Counter(words)

def similarity_sent(text1,text2):
    ###Removing stop words
    rm_stp_sent1=removeStopWords(text1)
    rm_stp_sent2=removeStopWords(text2)
    
    ##### Stemming
    stemming_sent1=stemma(rm_stp_sent1)
    stemming_sent2=stemma(rm_stp_sent2)
    
    ###Text to Vector
    sent_vector1=text_to_vector(stemming_sent1)
    sent_vector2=text_to_vector(stemming_sent2)
    
    ### calculating cosine similarity
    similarity_result=get_cosine(sent_vector1,sent_vector2)
    return(similarity_result)
import pandas as pd
def findingSimilarSentence(sentence1):
    df =pd.read_excel('Audit_Questions.xlsx')
    #
    #retriveQuesDatabase()
    #pd.read_excel('Audit_Questions.xlsx')
    
    cluster_sentence=[]
    #result=similarity(sentence1,sentence2)
    #sentence="What is the condition applied for field 'Real Position'?"
    colNames=['Sentence','Score']
    result_dataframe=pd.DataFrame(columns = colNames)
    for sent in df['aq_question']:
        result=similarity_sent(sentence1,sent)
        if result >0.8:
            #result_dataframe['Sentence'].append(str(sent))
            #result_dataframe['Score'].append(result)
            result_dataframe = result_dataframe.append({'Sentence':sent, 'Score':round(result,2)}, ignore_index=True)
            #cluster_sentence.append(sent)
    return(result_dataframe)
    #return(result)
def retriveQuesDatabase():
    try:
        conn = psycopg2.connect(database = "postgres", user = "postgres", password = "devpriya", host = "127.0.0.1", port = "5432")
        cur = conn.cursor()
        cur.execute("SELECT aq_id, aq_question,aq_createddate,aq_createdbyuserid,aq_host  from sfdc.tbl_auditquestions")
        rows = cur.fetchall()
        dataframe_db=pd.DataFrame(rows,columns=['aq_id','aq_question','aq_createddate','aq_createdbyuserid','aq_host'])
        return dataframe_db
    except DatabaseError:
        transaction.rollback()


# In[60]:

from flask import Flask, flash, redirect, render_template, request, session, abort
DEBUG = True
app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='APIs for Python Functions', validate=False)
ns = api.namespace('Sentence Similarity', 'Returns a list of Similar Sentences')
app.config['SECRET_KEY'] = '7d123427d441f27567d441f2b6176a'
start_time=0

port = int(os.getenv('PORT', 8080))

@app.route("/", methods=['GET', 'POST'])
def index():
     
    return render_template(
        'index1.html',form=request.form)

@app.route("/similar", methods=['POST'])
def similar():
    start_time= time.time()
    print("in the similar method",request.method)
    sentence_1=request.form['sentence1']
    print("sentence_1",sentence_1)
    result=findingSimilarSentence(sentence_1)
    result = result.sort_values(by=['Score'], ascending=[False])
    new_top_ten=result.drop_duplicates(['Score'], keep='first')
    top_five=new_top_ten.head(10)
    #new_top_ten=top_five.drop_duplicates(['Score'], keep='first')
    top_final=zip(top_five['Sentence'], top_five['Score']) #subset=['A', 'B'],
    elapsed_time = time.time() - start_time
    print(elapsed_time)
    return render_template(
        'result_list.html',your_list=top_final,titles = ['Sentence', 'Score'])
  
    #return render_template(
        #'result_list.html',your_list=[top_five.to_html(classes='male')],titles = ['Sentence', 'Score'])
  
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=False)


# In[ ]:



