import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection

from PIL import Image

#Set title
st.title('Total Data Science')

def main():
    activities = ['EDA','Visualization','model','About us']
    option = st.sidebar.selectbox('Selection option:',activities)

##EDA    
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')
        
        data = st.file_uploader('Upload file',type = ['csv','xlsx','txt'])
        st.success('Data loaded successfully')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            
            if st.checkbox('Display Shape'):
                st.write(df.shape)
            if st.checkbox('Display Columns'):
                st.write(df.columns)
            if st.checkbox('Select multiple columns'):
                selected_columns = st.multiselect('Select prefered columns: ',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            
            if st.checkbox('Display Summary'):
                st.write(df.describe().T)
            
            if st.checkbox('Display Null'):
                st.write(df.isnull().sum())
                
            if st.checkbox('Display datatype'):
                st.write(df.dtypes)
                
            if st.checkbox('Correlation'):
                st.write(df.corr())

##Visualization                
    
    elif option == 'Visualization':
        st.subheader('Data Visualization')
        
        data = st.file_uploader('Upload file',type = ['csv','xlsx','txt'])
        st.success('Data loaded successfully')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            
            if st.checkbox('Select multiple columns to plot'):
                selected_columns = st.multiselect('Your preffered columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
                
            if st.checkbox('Display heatmap'):
                st.write(sns.heatmap(df1.corr(), vmax=1, annot = True))
                st.pyplot()
                
            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1, diag_kind='kde'))
                st.pyplot()
                
            if st.checkbox('Display Piechart'):
                all_columns=df.columns.to_list()
                pie_columns=st.selectbox('select the column',all_columns)
                pieChart=df[pie_columns].value_counts().plot.pie()
                st.write(pieChart)
                st.pyplot()
                
#Model building
    elif option=='model':
        st.subheader('Model Building')
        
        data = st.file_uploader('Upload file',type = ['csv','xlsx','txt'])
        st.success('Data loaded successfully')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            
            
            if st.checkbox('Select multiple columns'):
                new_data = st.multiselect('Selected columns',df.columns)
                df1 = df[new_data]
                st.dataframe(df1)
                x=df1.iloc[:,0:-1]
                y=df1.iloc[:,-1]
                
                
                
            seed=st.sidebar.slider('Seed',1,200)
            
            classifier_name=st.sidebar.selectbox('Select Algo',('KNN','SVM','Logistic','DT'))
            
            def add_parameter(name_of_clf):
                param=dict()
                if name_of_clf == 'SVM':
                    C=st.sidebar.slider('C',0.01,10.0)
                    param['C']=C
                if name_of_clf == 'KNN':
                    K=st.sidebar.slider('K',1,10)
                    param['K']=K
                    return param
                
            params = add_parameter(classifier_name)
            
            #Define function for classfier
            def get_classifier(name_of_clf,params):
                clf=None
                if name_of_clf == 'SVM':
                    clf=SVC(C=params['C'])
                elif name_of_clf == 'KNN':
                    clf = KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_clf == 'Logistic':
                    clf = LogisticRegression()
                elif name_of_clf == 'DT':
                    clf = DecisionTreeClassifier()
                return clf
                

            clf = get_classifier(classifier_name,params)

            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=seed)
            
            clf.fit(x_train,y_train)
            
            y_pred=clf.predict(x_test)
            st.write('Predicted values',y_pred)
            
            accuracy=accuracy_score(y_test,y_pred)
            
            st.write('Name of classifier ',classifier_name)
            st.write('Accuracy is ', accuracy)
                
            
        
            
            
            
            
        
        
    
            
        






if __name__ == '__main__':
    main()