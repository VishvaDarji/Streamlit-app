# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:15:58 2021

@author: vishwa darji
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd

gbr_model=pickle.load(open('gbr_model.pkl','rb'))

rfr_model=pickle.load(open('rfr_model.pkl','rb'))
rfr_bone=pickle.load(open('rfr_bone.pkl','rb'))
rfr_bclSwb=pickle.load(open('rfr_bclSwb.pkl','rb'))

def mean(lst):
    return sum(lst) / len(lst)
def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]
def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs
@st.cache
def predict_age(file,tissue_type):
    if 'csv' in file.name:
        df=pd.read_csv(file)
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)

    elif 'txt' in file.name:
        #CLEANING THE INPUT RAW FILE
        input_df=pd.read_csv(file,sep='  ') #splitting with \n
        input_df1=input_df.iloc[:, 0].str.split('\t').tolist()
        input_df1=pd.DataFrame(input_df1)

        input_df1.set_index(input_df1.iloc[:,0],inplace=True,drop=True) #setting index
        input_df1.drop(input_df1[[0]],axis=1,inplace=True)

        #slicing first 2 cols out
        input_df1=input_df1.iloc[:,0:-1]

        #removing extra rows at the bottom
        input_df1.index = input_df1.index + input_df1.groupby(level=0).cumcount().astype(str).replace('0','')
        drop1=input_df1[input_df1.index.str.startswith('ch')].index.tolist()
        drop2=input_df1[input_df1.index.str.startswith('rs')].index.tolist()
        input_df1.drop(drop1,axis=0, inplace=True)
        input_df1.drop(drop2,axis=0, inplace=True)

        #remving extra rows from the top before ID_REF row
        index_series=pd.Series(input_df1.index)
        i1=index_series.tolist().index('ID_REF')
        del_index=index_series[:i1].tolist()
        input_df1.drop(del_index,inplace=True)

        #resetting index
        input_df1.reset_index(inplace=True)

        #renaming col names
        input_df1.rename(columns=input_df1.iloc[0],inplace=True)
        input_df1.drop([0],inplace=True)

        input_df1.set_index(input_df1.iloc[:,0],inplace=True,drop=True) #setting index
        input_df1.drop('ID_REF',axis=1,inplace=True)

        #converting it to float
        input_df1=input_df1.astype(np.float64)
        df=input_df1.copy()

    if tissue_type=='Leukocytes':
        #feature file loading
        leu_feat=pd.read_csv('Top400_leu_feature.csv')
        leu_feat.set_index(leu_feat.iloc[:,0],inplace=True)
        leu_feat.drop('Unnamed: 0',axis=1,inplace=True)

        pred_leu=[]
        for feature in leu_feat.index:
            if feature in df.index:
                pred_leu.append(df.loc[feature])
            else:
                pred_leu.append(0)
        idxs=list_duplicates_of(pred_leu,0)
        for i in idxs:
            pred_leu[i]=mean(pred_leu)
        if len(pred_leu)==0:
            st.write('Input file does not have any feature in common with model features')
        
        return gbr_model.predict(np.array(pred_leu).reshape(1,-1))
        pred_leu.clear()

    elif tissue_type=='Whole blood':
        #feature file loading
        blood_feat=pd.read_csv('Top340_blood_feature.csv')
        blood_feat.set_index(blood_feat.iloc[:,0],inplace=True)
        blood_feat.drop('Unnamed: 0',axis=1,inplace=True)

        pred_blood=[]
        for feature in blood_feat.index:
            if feature in df.index:
                pred_blood.append(df.loc[feature])
            else:
                pred_blood.append(0)
        idxs=list_duplicates_of(pred_blood,0)
        for i in idxs:
            pred_blood[i]=mean(pred_blood)
        if len(pred_blood)==0:
            st.write('Input file does not have any feature in common with model features')
        
        return rfr_model.predict(np.array(pred_blood).reshape(1,-1))
        pred_blood.clear()

    elif tissue_type=='Bone':
        #feature file loading
        bone_feat=pd.read_csv('Top29_bone_feature.csv')
        bone_feat.set_index(bone_feat.iloc[:,0],inplace=True)
        bone_feat.drop('Unnamed: 0',axis=1,inplace=True)

        pred_bone=[]
        for feature in bone_feat.index:
            if feature in df.index:
                pred_bone.append(df.loc[feature])
            else:
                pred_bone.append(0)
        idxs=list_duplicates_of(pred_bone,0)
        for i in idxs:
            pred_bone[i]=mean(pred_bone)
        if len(pred_bone)==0:
            st.write('Input file does not have any feature in common with model features')
        
        return rfr_bone.predict(np.array(pred_bone).reshape(1,-1))
        pred_bone.clear()

    elif tissue_type=='Buccal swab':
        #feature file loading
        bs_feat=pd.read_csv('Top60_bs_feature.csv')
        bs_feat.set_index(bs_feat.iloc[:,0],inplace=True)
        bs_feat.drop('Unnamed: 0',axis=1,inplace=True)

        pred_bs=[]
        for feature in bs_feat.index:
            if feature in df.index:
                pred_bs.append(df.loc[feature])
            else:
                pred_bs.append(0)
        idxs=list_duplicates_of(pred_bs,0)
        for i in idxs:
            pred_bs[i]=mean(pred_bs)
        if len(pred_bs)==0:
            st.write('Input file does not have any feature in common with model features')
        
        return rfr_bclSwb.predict(np.array(pred_bs).reshape(1,-1))
        pred_bs.clear()

def main():
    #adding sidebar
    st.sidebar.header('MENU')
    menu=['HOME','AGE PREDICTOR','ABOUT','CONTACT DETAILS']
    choice=st.sidebar.selectbox('Navigate',menu)

    if choice=='HOME':
        st.title('''WELCOME TO MY STREAMLIT APPLICATION\n
FOR AGE PREDICTION''')
        st.text('Navigate to the Age predictor page through the sidebar')
        st.image('home_app.png',width=280)
    
    elif choice=='AGE PREDICTOR':
        st.title('STREAMLIT AGE PREDICTOR APP')
        html_temp = """
        <div style="background-color:tomato;padding:5px">
        <h2 style="color:white;text-align:center;"> Age Predictor </h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        #input taking
        st.write('\n\n')
        file=st.file_uploader('Upload the sample for prediction',type=['txt','csv'])
        if file:
            tissue_type=st.selectbox('Select the tissue type', ('Leukocytes','Whole blood','Bone','Buccal swab'))

        st.subheader('OR Use the tissue specific sample files:')
        sample_file=st.selectbox('Choose the model',['None','Leukocytes','Whole blood','Bone','Buccal swab'])
        if sample_file=='None':
            pass
        elif sample_file=='Leukocytes':
            leu_sample=st.selectbox('Select sample files for leukocyte model',['Leukocytes_test(20).csv','Leukocytes_test(48).csv'])
        elif sample_file=='Whole blood':
            wb_sample=st.selectbox('Select sample file for whole blood model',['WholeBlood_test(31).csv','WholeBlood_test(47).csv'])
        elif sample_file=='Bone':
            bone_sample=st.selectbox('Select sample file for bone model',['Bone_test(28.66).csv','Bone_test(16.51).csv'])
        elif sample_file=='Buccal swab':
            bs_sample=st.selectbox('Select sample file for buccal swab model',['buccalSwab_test(18).csv','buccalSwab_test(18.08).csv']) 
       
        #processing input_df1t
        if st.button('Predict age'):
            if file:
                outputFOR_input_file=predict_age(file,tissue_type)
                st.success('Predicted Age {}'.format(outputFOR_input_file))
            elif sample_file: 
                if sample_file=='Leukocytes':
                    outputFOR_LEUsample_file=leu_age(leu_sample)
                    st.success('Predicted Age {}'.format(outputFOR_LEUsample_file))
                if sample_file=='Whole blood':
                    outputFOR_WBsample_file=wb_age(wb_sample)
                    st.success('Predicted Age {}'.format(outputFOR_WBsample_file))
                if sample_file=='Bone':
                    outputFOR_BONEsample_file=bone_age(bone_sample)
                    st.success('Predicted Age {}'.format(outputFOR_BONEsample_file))
                if sample_file=='Buccal swab':
                    outputFOR_BSsample_file=bs_age(bs_sample)
                    st.success('Predicted Age {}'.format(outputFOR_BSsample_file))
    elif choice=='ABOUT':
        st.write('**OBJECTIVE: ** It is mainly to learn how to build a ML web application and then deploy it')
        st.write('This app is also a part of my last semester project at undergraduation level')
    elif choice=='CONTACT DETAILS':
        st.write('''\n
            ***Name***: Vishva Darji\n
            ***Linkedin profile:*** www.linkedin.com/in/vishva-darji\n 
            ***E-mail ID:*** vishvadarji2832001@gmail.com''')

global leu_sample, wb_sample, bone_sample, bs_sample
@st.cache
def leu_age(leu_sample):
    leu_feat=pd.read_csv('Top400_leu_feature.csv')
    leu_feat.set_index(leu_feat.iloc[:,0],inplace=True)
    leu_feat.drop('Unnamed: 0',axis=1,inplace=True)
    if leu_sample=='Leukocytes_test(20).csv':
        df=pd.read_csv('Leukocytes_test(20).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_leu=[]
        for feature in leu_feat.index:
            if feature in df.index:
                pred_leu.append(df.loc[feature])
            else:
                pred_leu.append(0)
        idxs=list_duplicates_of(pred_leu,0)
        for i in idxs:
            pred_leu[i]=mean(pred_leu)
        if len(pred_leu)==0:
            st.write('Input file does not have any feature in common with model features')
        return gbr_model.predict(np.array(pred_leu).reshape(1,-1))
        pred_leu.clear()
    elif leu_sample=='Leukocytes_test(48).csv':
        df=pd.read_csv('Leukocytes_test(48).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_leu=[]
        for feature in leu_feat.index:
            if feature in df.index:
                pred_leu.append(df.loc[feature])
            else:
                pred_leu.append(0)
        idxs=list_duplicates_of(pred_leu,0)
        for i in idxs:
            pred_leu[i]=mean(pred_leu)
        if len(pred_leu)==0:
            st.write('Input file does not have any feature in common with model features')
    
        return gbr_model.predict(np.array(pred_leu).reshape(1,-1))
        pred_leu.clear() 
@st.cache
def wb_age(wb_sample):
    blood_feat=pd.read_csv('Top340_blood_feature.csv')
    blood_feat.set_index(blood_feat.iloc[:,0],inplace=True)
    blood_feat.drop('Unnamed: 0',axis=1,inplace=True)
    if wb_sample=='WholeBlood_test(31).csv':
        df=pd.read_csv('WholeBlood_test(31).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_blood=[]
        for feature in blood_feat.index:
            if feature in df.index:
                pred_blood.append(df.loc[feature])
            else:
                pred_blood.append(0)
        idxs=list_duplicates_of(pred_blood,0)
        for i in idxs:
            pred_blood[i]=mean(pred_blood)
        if len(pred_blood)==0:
            st.write('Input file does not have any feature in common with model features')
        return rfr_model.predict(np.array(pred_blood).reshape(1,-1))
        pred_blood.clear()

    elif wb_sample=='WholeBlood_test(47).csv':
        df=pd.read_csv('WholeBlood_test(47).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_blood=[]
        for feature in blood_feat.index:
            if feature in df.index:
                pred_blood.append(df.loc[feature])
            else:
                pred_blood.append(0)
        idxs=list_duplicates_of(pred_blood,0)
        for i in idxs:
            pred_blood[i]=mean(pred_blood)
        if len(pred_blood)==0:
            st.write('Input file does not have any feature in common with model features')
        return rfr_model.predict(np.array(pred_blood).reshape(1,-1))
        pred_blood.clear()
@st.cache
def bone_age(bone_sample):
    bone_feat=pd.read_csv('Top29_bone_feature.csv')
    bone_feat.set_index(bone_feat.iloc[:,0],inplace=True)
    bone_feat.drop('Unnamed: 0',axis=1,inplace=True)
    if bone_sample=='Bone_test(28.66).csv':
        df=pd.read_csv('Bone_test(28.66).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_bone=[]
        for feature in bone_feat.index:
            if feature in df.index:
                pred_bone.append(df.loc[feature])
            else:
                pred_bone.append(0)
        idxs=list_duplicates_of(pred_bone,0)
        for i in idxs:
            pred_bone[i]=mean(pred_bone)
        if len(pred_bone)==0:
            st.write('Input file does not have any feature in common with model features')
        return rfr_bone.predict(np.array(pred_bone).reshape(1,-1))
        pred_bone.clear()

    elif bone_sample=='Bone_test(16.51).csv':
        df=pd.read_csv('Bone_test(16.51).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_bone=[]
        for feature in bone_feat.index:
            if feature in df.index:
                pred_bone.append(df.loc[feature])
            else:
                pred_bone.append(0)
        idxs=list_duplicates_of(pred_bone,0)
        for i in idxs:
            pred_bone[i]=mean(pred_bone)
        if len(pred_bone)==0:
            st.write('Input file does not have any feature in common with model features')
        return rfr_bone.predict(np.array(pred_bone).reshape(1,-1))
        pred_bone.clear()
@st.cache
def bs_age(bs_sample):
    bs_feat=pd.read_csv('Top60_bs_feature.csv')
    bs_feat.set_index(bs_feat.iloc[:,0],inplace=True)
    bs_feat.drop('Unnamed: 0',axis=1,inplace=True)
    if bs_sample=='buccalSwab_test(18).csv':
        df=pd.read_csv('buccalSwab_test(18).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_bs=[]
        for feature in bs_feat.index:
            if feature in df.index:
                pred_bs.append(df.loc[feature])
            else:
                pred_bs.append(0)
        idxs=list_duplicates_of(pred_bs,0)
        for i in idxs:
            pred_bs[i]=mean(pred_bs)
        if len(pred_bs)==0:
            st.write('Input file does not have any feature in common with model features')
        return rfr_bclSwb.predict(np.array(pred_bs).reshape(1,-1))
        pred_bs.clear()

    elif bs_sample=='buccalSwab_test(18.08).csv':
        df=pd.read_csv('buccalSwab_test(18.08).csv')
        df.set_index(df.iloc[:,0],inplace=True)
        df.drop('ID_REF',axis=1,inplace=True)
        pred_bs=[]
        for feature in bs_feat.index:
            if feature in df.index:
                pred_bs.append(df.loc[feature])
            else:
                pred_bs.append(0)
        idxs=list_duplicates_of(pred_bs,0)
        for i in idxs:
            pred_bs[i]=mean(pred_bs)
        if len(pred_bs)==0:
            st.write('Input file does not have any feature in common with model features')
        return rfr_bclSwb.predict(np.array(pred_bs).reshape(1,-1))
        pred_bs.clear()


if __name__=='__main__':
    main()
