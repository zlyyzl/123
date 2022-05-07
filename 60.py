#!/usr/bin/env python
# coding: utf-8

# In[16]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from xgboost import XGBClassifier

# In[17]:


model = pickle.load(open( "tuned_xgboostz4.p", "rb" ))


# In[18]:


def transform(data):    
        data['Collateral_status'] = pd.factorize(data['Collateral_status'])[0]
        return data


# In[19]:


def predict(model, input_df):
    transform_data = transform(input_df)
    predictions_df = model.predict(transform_data)
    predictions = predictions_df[0]
    return predictions_df


# In[20]:


def run():


 st.title('Functional outcome prediction  App for patients with  anterior circulation large vessel occlusion after mechanical thrombectomy')

add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online_number","Batch", "Online_slide"))

# In[21]:


image = Image.open('hospital1.webp')
st.image(image, use_column_width=True)


# In[22]:



# In[23]:


import streamlit as st
import openpyxl
import base64
csv_exporter=openpyxl.Workbook()
sheet=csv_exporter.active
sheet.cell(row=1,column=1).value='Age'
sheet.cell(row=1,column=2).value='eGFR'
sheet.cell(row=1,column=3).value='NIHSS'
sheet.cell(row=1,column=4).value='Albumin'
sheet.cell(row=1,column=5).value='Albumin-to-globulin_ratio'
sheet.cell(row=1,column=6).value='Serum_creatinine'
sheet.cell(row=1,column=7).value='White_blood_cell_count'
sheet.cell(row=1,column=8).value='Blood_neutrophils_count'
sheet.cell(row=1,column=9).value='Fasting_blood_glucose'
sheet.cell(row=1,column=10).value='Collateral_status'
csv_exporter.save('for predictions.csv')#注意！文件此时保存在内存中且为字节格式文件
data=open('for predictions.csv','rb').read()#以只读模式读取且读取为二进制文件
b64 = base64.b64encode(data).decode('UTF-8')#解码并加密为base64
href = f'<a href="data:file/data;base64,{b64}" download="myresults.csv">Download csv file</a>'#定义下载链接，默认的下载文件名是myresults.xlsx
st.markdown(href, unsafe_allow_html=True)#输出到浏览器
csv_exporter.close()


# In[24]:


if add_selectbox == 'Online_number':
      st.write('This is a web app to predict your 90-day functional outcome based on several features. please fill in the blanks with corresponding data. After that,click on the Predict button at the bottom to see the prediction of the classifier. ')
    
      Age = st.number_input('Age', min_value = 18,max_value = 95 ,value = 73) 
      eGFR = st.number_input('eGFR', min_value = 10.00,max_value = 250.00,value = 111.5)
      NIHSS = st.number_input('NIHSS', min_value = 4,max_value = 38,value = 30)
      Albumin = st.number_input('Albumin', min_value = 0.20,max_value = 80.00,value = 39.20)
      Albumin_to_globulin_ratio = st.number_input('Albumin-to-globulin_ratio', min_value = 0.50,max_value = 10.00 , value = 1.50)
      Serum_creatinine = st.number_input('Serum_creatinine', min_value = 10.00,max_value = 700.00,value = 50.00)
      White_blood_cell_count = st.number_input('White_blood_cell_count', min_value = 3.00, max_value = 22.00 ,value = 8.5)
      Blood_neutrophils_count = st.number_input('Blood_neutrophils_count', min_value = 2.00, max_value = 20.00 ,value = 7.40)
      Fasting_blood_glucose = st.number_input('Fasting_blood_glucose', min_value = 2.50, max_value = 25.00 , value = 11.78)
      Collateral_status = st.selectbox('Collateral_status', [1, 0])

      output=""
 

      features  = {'Age': Age, 'eGFR': eGFR,
               'NIHSS': NIHSS, 'Albumin': Albumin,
               'Albumin-to-globulin_ratio': Albumin_to_globulin_ratio,'Serum_creatinine': Serum_creatinine, 
               'White_blood_cell_count': White_blood_cell_count,'Blood_neutrophils_count': Blood_neutrophils_count, 
               'Fasting_blood_glucose': Fasting_blood_glucose,  'Collateral_status': Collateral_status}
      print(features)


      features_df = pd.DataFrame([features])
      print(features_df)

        
      if st.button('Predict'): 
            output = predict(model, features_df) 
      st.write(' Based on feature values, your functional outcome is '+ str(output))


# In[9]:


if add_selectbox == 'Batch':
        st.write('This is a web app to predict your 90-day functional outcome based on several features. Please click on the link to download the form and fill in the corresponding data. After that, click on the Browse files button to upload file for prediciton, you can see the prediction of the classifier at the bottom.This app can predict the prognosis of multiple patients at one time. ')


        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        
        if file_upload is not None:
            data = pd.read_csv(file_upload,sep=',')                       
            predictions = predict(model,data) 
            predictions = pd.DataFrame(predictions,columns = ['Predictions'])
            st.write(predictions)
          

# In[12]:


if add_selectbox == 'Online_slide':
    
      st.write('This is a web app to predict your 90-day functional outcome based on several features that you can see in the sidebar. Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction of the classifier.')

      Age = st.sidebar.slider(label = 'Age', min_value = 18,
                          max_value = 95 ,
                          value = 70,
                          step = 1) 
      eGFR = st.sidebar.slider(label = 'eGFR', min_value = 10.00,
                          max_value = 250.00,
                          value = 70.00,
                           step = 0.10)
      NIHSS = st.sidebar.slider(label = 'NIHSS', min_value = 4,
                          max_value = 38,
                          value = 30,
                          step = 1)
      Albumin = st.sidebar.slider(label = 'Albumin', min_value = 0.20,
                          max_value = 80.00,
                          value = 40.00,
                          step = 0.10)
      Albumin_to_globulin_ratio = st.sidebar.slider(label = 'Albumin-to-globulin_ratio', min_value = 0.50,
                          max_value = 10.00 ,
                          value = 1.00,
                          step = 0.10)
      Serum_creatinine = st.sidebar.slider(label = 'Serum_creatinine', min_value = 2.50,
                          max_value = 700.50 ,
                          value = 12.50,
                          step = 0.10)
      White_blood_cell_count = st.sidebar.slider(label = 'White_blood_cell_count', min_value = 3.00,
                          max_value = 22.00 ,
                          value = 16.00,
                          step = 0.10)
      Blood_neutrophils_count = st.sidebar.slider(label = 'Blood_neutrophils_count', min_value = 2.00,
                          max_value = 20.00 ,
                          value = 4.00,
                          step = 0.01)
      Fasting_blood_glucose = st.sidebar.slider(label = 'Fasting_blood_glucose', min_value = 2.50,
                          max_value = 25.00 ,
                          value = 10.00,
                          step = 0.10)   
      Collateral_status = st.sidebar.selectbox('Collateral_status', [1, 0])

      output=""
  

      features  = {'Age': Age, 'eGFR': eGFR,
               'NIHSS': NIHSS, 'Albumin': Albumin,
               'Albumin-to-globulin_ratio': Albumin_to_globulin_ratio,'Serum_creatinine': Serum_creatinine, 
               'White_blood_cell_count': White_blood_cell_count,'Blood_neutrophils_count': Blood_neutrophils_count, 
               'Fasting_blood_glucose': Fasting_blood_glucose,  'Collateral_status': Collateral_status}
      print(features)

      features_df = pd.DataFrame([features])
      st.table(features_df)
        
      if st.button('Predict'): 
           output = predict(model=model, input_df=features_df) 
      st.write(' Based on feature values, your functional outcome is '+ str(output))


# In[13]:


if __name__ == '__main__':
    run()



# In[ ]:




