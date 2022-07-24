#!/usr/bin/env python
# coding: utf-8

# # AI in Enterprise Systems (AIDI 2004-02)

# ## Lab Assignment 4 - Flask API

# #### Done by:- Abraham Mathew (100829875)

# ### Flask API

# In[10]:


import flask
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = flask.Flask(__name__)
model = pickle.load(open('lab4_model.pkl','rb'))

@app.route('/')
def main():
    return(render_template('main.html'))

@app.route('/Predict',method = ['POST'])
def predict():
    init_features = [float(x) for x in reequest.form.values()]
    final = np.array(init_features)
    prediction = model.predict(final)
    return render_template('main.html', prediction_text = 'The fish belongs to the Species {}'.format(str(prediction)))
    
if __name__=='__main__':
    app.run()


# In[ ]:




