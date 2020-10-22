import streamlit as st
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve
from sklearn.metrics import precision_score,recall_score 
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures

from datetime import date 
from datetime import timedelta 

def main():
    st.title("predict covid-19 cases")
    st.sidebar.title("predict covid-19 cases")
    st.markdown("analyse different ML algorithms here")
    st.sidebar.title("analyse different ML algorithms here")

    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv('./world-ds.csv',error_bad_lines=False,parse_dates=["date"])
        data = data[data.location == 'India']
        #data.head()
        #label=LabelEncoder()
        #for col in data.columns:
            #data[col]=label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(data):
        df1=pd.DataFrame()
        df1["year"]=data.date.dt.year
        df1["month"]=data.date.dt.month
        df1["day"]=data.date.dt.day
        y=data.new_cases
        x=df1

        l=[]
        day=[]
        month=[]
        year=[]
        today = date.today() 
        for i in range(0,15):
            l.append(today + timedelta(days = i))
            day.append(l[i].day)
            month.append(l[i].month)
            year.append(l[i].year)

        df2=pd.DataFrame({"year":year,"month":month,"day":day})    
        #x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        #return x_train,x_test,y_train,y_test
        return df1,df2,y,l

    df=load_data()
    df=df.dropna()
    x,x1,y,l=split(df) 
    st.sidebar.subheader("choose classifier")
    classifier = st.sidebar.selectbox("classifier",("svr","linear regression","random forest","ada boost")) 
    metrics = st.sidebar.multiselect("what do u want to check?", ('future 10 days plot','future 10 days cases','performance on trained data','plot on train data'))     

    
    def plot_metrics(metrics,model,str,accuracy):
        if 'future 10 days plot' in metrics:
            st.subheader("future 10 days plot")
            fig, ax1 = plt.subplots()
            ax1.plot(l,model.predict(x1),label=str,color='orange')
            ax1.legend()
            st.pyplot(fig)

        if 'plot on train data' in metrics:
            st.subheader("trained and predicted plot")
            fig, ax1 = plt.subplots()
            ax1.plot(df.date,model.predict(x),label=str,color='orange')
            ax1.plot(df.date, df.new_cases)
            ax1.legend()
            st.pyplot(fig)

        if 'future 10 days cases' in metrics:
            st.subheader("future 10 days cases")
            #for i in range(0,10):
            st.write(model.predict(x1))  

        if 'performance on trained data' in metrics:
            st.subheader("performance on trained data")
            st.write("accuracy: ",accuracy.round(2))     
    
    
    if st.sidebar.button("classify",key='classify'):
        if classifier=="svr":
            modelsvr = SVR().fit(x,y)
            accuracy=modelsvr.score(x,y)
            y_pred=modelsvr.predict(x)
            #st.write("accuracy: ",accuracy.round(2))
            #st.write("precision: ",precision_score(y_pred,y).round(2))
            #st.write("recall: ",recall_score(y,y_pred).round(2))
            plot_metrics(metrics,modelsvr,classifier,accuracy)

        if classifier=="linear regression":
            modellr = LinearRegression().fit(x, y)    
            accuracy=modellr.score(x,y)
            y_pred=modellr.predict(x)
            #st.write("accuracy: ",accuracy.round(2))
            #st.write("precision: ",precision_score(y_pred,y).round(2))
            #st.write("recall: ",recall_score(y,y_pred).round(2))
            plot_metrics(metrics,modellr,classifier,accuracy)

        if classifier=="random forest":
            model = RandomForestRegressor()
            model.fit(x, y)
            accuracy=model.score(x,y)
            y_pred=model.predict(x)
            #st.write("accuracy: ",accuracy.round(2))
            #st.write("precision: ",precision_score(y_pred,y).round(2))
            #st.write("recall: ",recall_score(y,y_pred).round(2))
            plot_metrics(metrics,model,classifier,accuracy)

        if classifier=="ada boost":
            modelab = AdaBoostRegressor().fit(x, y)
            accuracy=modelab.score(x,y)
            y_pred=modelab.predict(x)
            #st.write("accuracy: ",accuracy.round(2))
            #st.write("precision: ",precision_score(y_pred,y).round(2))
            #st.write("recall: ",recall_score(y,y_pred).round(2))
            plot_metrics(metrics,modelab,classifier,accuracy) 






if __name__ == '__main__':
    main()    