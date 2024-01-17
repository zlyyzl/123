#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
shap.initjs()
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from pycaret import classification
from pycaret.classification import setup, get_logs,create_model, predict_model, tune_model, save_model,load_model


# In[2]:


import streamlit.components.v1 as components


# In[3]:


if st.button("Clique aqui para baixar a base de dados"):
     download_base_dados()
     # Gere um link para o arquivo Excel gerado
     with open("Excel_model_societal.xlsm", "rb") as file:
         b64 = base64.b64encode(file.read()).decode()
         href = f'<a href="data:application/octet-stream;base64,{b64}" download="base_dados.xlsx">Download da base de dados</a>'
     st.markdown(href, unsafe_allow_html=True)


# In[4]:


if __name__ == '__main__':
    run()


# In[ ]:




