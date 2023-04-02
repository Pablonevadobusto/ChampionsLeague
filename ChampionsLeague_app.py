import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import keras
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from keras.utils import pad_sequences
from sklearn import preprocessing

## set page configuration
st.set_page_config(page_title= 'Champions League', layout='wide')

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]  > .main{{
background-image: url("https://w0.peakpx.com/wallpaper/355/120/HD-wallpaper-champions-league-icio-uefa-champions-league.jpg");
background-size: cover;
background-position: top-left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background-color: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

audio_file = open('UEFA Champions League Anthem.mp3','rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')

## Loading the ann model
### Opening a file with Pickle (Logistic Regression model was saved as Pickle (binary) format)
with open('knn_model.pkl', 'rb') as file:
          model_knn = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
          model_LR = pickle.load(file)
## load the copy of the dataset
HomeStats = pd.read_csv('HomeStats.csv')
HomeStats = HomeStats.iloc[:,1:]  # because in the column 0 there was a bizarre column called "Unnamed: 0" that seamed to be indexes
AwayStats = pd.read_csv('AwayStats.csv')
AwayStats = AwayStats.iloc[:,1:] 

# ## set page configuration
# st.set_page_config(page_title= 'Champions League', layout='wide')

## add page title and content
st.title('Identifying Champions League winner using KNN and Logistic Regression')
st.write('''
 Champions League winner Prediction App.

 This app predicts the football Champions League winner using a KNN for multi-class problem and Logistic Regression for binary problem.
''')


# add image
col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    image = Image.open("ChampionsLeagueImage3.png")
    st.image(image, width = 800)
with col3:
    st.write("")

teamlist = list(HomeStats['HomeClubName'].unique())
teamlist.sort()
## get user imput
#email_text = st.text_input('Email Text:')
st.sidebar.header('Input football clubs')
teamsGroupA = []
teamsGroupB = []
teamsGroupC = []
teamsGroupD = []
teamsGroupE = []
teamsGroupF = []
teamsGroupG = []
teamsGroupH = []



def user_input_GroupA():
    global teamlist
    A = st.sidebar.multiselect("Group A:", options = teamlist)
    teamsGroupA.extend(A)
    teamlist = [i for i in teamlist if i not in A]
    
    GroupA = pd.DataFrame({'A':teamsGroupA})
    return GroupA

gpA = user_input_GroupA()

def user_input_GroupB():
    global teamlist

    B = st.sidebar.multiselect("Group B:", options = teamlist)
    teamsGroupB.extend(B)
    teamlist = [i for i in teamlist if i not in B]

    
    GroupB = pd.DataFrame({'B':teamsGroupB})
    return GroupB

gpB = user_input_GroupB()

def user_input_GroupC():
    global teamlist
    C = st.sidebar.multiselect("Group C:", options = teamlist)
    teamsGroupC.extend(C)
    teamlist = [i for i in teamlist if i not in C]
    
    GroupC = pd.DataFrame({'C':teamsGroupC})
    return GroupC

gpC = user_input_GroupC()

def user_input_GroupD():
    global teamlist
    D = st.sidebar.multiselect("Group D:", options = teamlist)
    teamsGroupD.extend(D)
    teamlist = [i for i in teamlist if i not in D]
    
    GroupD = pd.DataFrame({'D':teamsGroupD})
    return GroupD

gpD = user_input_GroupD()

def user_input_GroupE():
    global teamlist
    E = st.sidebar.multiselect("Group E:", options = teamlist)
    teamsGroupE.extend(E)
    teamlist = [i for i in teamlist if i not in E]
    
    GroupE = pd.DataFrame({'E':teamsGroupE})
    return GroupE

gpE = user_input_GroupE()

def user_input_GroupF():
    global teamlist
    F = st.sidebar.multiselect("Group F:", options = teamlist)
    teamsGroupF.extend(F)
    teamlist = [i for i in teamlist if i not in F]
    
    GroupF = pd.DataFrame({'F':teamsGroupF})
    return GroupF

gpF = user_input_GroupF()

def user_input_GroupG():
    global teamlist
    G = st.sidebar.multiselect("Group G:", options = teamlist)
    teamsGroupG.extend(G)
    teamlist = [i for i in teamlist if i not in G]
    
    GroupG = pd.DataFrame({'G':teamsGroupG})
    return GroupG

gpG = user_input_GroupG()

def user_input_GroupH():
    global teamlist
    H = st.sidebar.multiselect("Group H:", options = teamlist)
    teamsGroupH.extend(H)
    teamlist = [i for i in teamlist if i not in H]
    
    GroupH = pd.DataFrame({'H':teamsGroupH})
    return GroupH

gpH = user_input_GroupH()


st.write(gpA.T)
st.write(gpB.T)
st.write(gpC.T)
st.write(gpD.T)
st.write(gpE.T)
st.write(gpF.T)
st.write(gpG.T)
st.write(gpH.T)

import itertools
table = pd.DataFrame()
GroupRange = {'A':teamsGroupA,'B': teamsGroupB,'C':teamsGroupC,'D':teamsGroupD,'E':teamsGroupE,'F':teamsGroupF,'G':teamsGroupG,'H':teamsGroupH}
for i in GroupRange:
    a = list(itertools.combinations(GroupRange[i], 2))
    a2 = [t[::-1] for t in a]
    a.extend(a2)
    table2 = pd.DataFrame(a, columns=['HomeTeam', 'AwayTeam'])
    table2.insert(0,'Group',i)
    table = table.append(table2)

try:
            
    st.subheader('GROUPS')

    table1=pd.merge(table, HomeStats,'left', left_on='HomeTeam',right_on='HomeClubName')
    clmatches=pd.merge(table1, AwayStats,'left',left_on='AwayTeam',right_on='AwayClubName')

    clmatches=clmatches.drop(['HomeClubName','AwayClubName'],axis=1)
    scaler = StandardScaler()

    scaled_feat=scaler.fit_transform(clmatches.iloc[:,3:])
    Xcl=pd.DataFrame(scaled_feat,columns = list(clmatches.iloc[:,3:].columns))

    predcl=model_knn.predict(Xcl)

    clmatches['Results']=predcl
    clresults=clmatches[['Group','HomeTeam','AwayTeam','Results']]
    clresults['Homepts']=0
    clresults['Awaypts']=0
    clresults['Homepts'][clresults['Results']==1]=3
    clresults['Awaypts'][clresults['Results']==2]=3
    clresults['Homepts'][clresults['Results']==0]=1
    clresults['Awaypts'][clresults['Results']==0]=1
    st.write(clresults)

    hpts = clresults[['Group','HomeTeam','Homepts']].groupby(['Group','HomeTeam']).sum().reset_index()
    apts = clresults[['Group','AwayTeam','Awaypts']].groupby(['Group','AwayTeam']).sum().reset_index()

    clpred = pd.concat([hpts, apts], axis=1)
    clpred['Totalpts'] = clpred['Homepts'] + clpred['Awaypts']
    clpred.drop(['Homepts','AwayTeam','Awaypts'],axis= 1, inplace=True)
    clpred = clpred.loc[:,~clpred.columns.duplicated()]
    clpred = clpred.groupby(['Group','HomeTeam']).sum()
    #clpred.sort_values(['Group','Totalpts'],ascending=False).groupby('Group')
    clpred = clpred.sort_values(['Group','Totalpts'], ascending=[True,False])

    st.write(clpred)

    cl_groups = clpred.sort_values(['Group','Totalpts'],ascending=[True,False]).groupby('Group').head(2)
    st.write(cl_groups)

    first = cl_groups.groupby('Group').head(1).reset_index()
    first = first[["HomeTeam"]]
    first.rename(columns={'HomeTeam':'Home'}, inplace=True)

    second = cl_groups.sort_values(['Group','Totalpts','HomeTeam'],ascending=[True,True,False]).groupby('Group').head(1).reset_index()
    second = second[["HomeTeam"]]
    second.rename(columns={'HomeTeam':'Away'}, inplace=True)

    second = second.reindex(np.random.permutation(second.index))
    second = second.reset_index().iloc[:,1:]

## KNOCK OUT
    st.subheader('KNOCK OUT:sunglasses:')

    if st.button('Predict'): 
        
        ## ROUND OF 16
        st.subheader(':red[Round of 16]:soccer:')
        roundof16matches = pd.concat([first,second],axis=1).sort_index()

        table1=pd.merge(roundof16matches, HomeStats,'left', left_on='Home',right_on='HomeClubName')
        roundof16total=pd.merge(table1, AwayStats,'left',left_on='Away',right_on='AwayClubName')

        roundof16total=roundof16total.drop(['HomeClubName','AwayClubName'],axis=1)

        ## Standard Scaler
        scaler.fit(roundof16total.iloc[:,2:])
        scaled_feat1=scaler.transform(roundof16total.iloc[:,2:])
        Xcl2=pd.DataFrame(scaled_feat1,columns = list(roundof16total.iloc[:,2:].columns))

                
        predcl2=model_LR.predict(Xcl2)

        roundof16total['Results']=predcl2
        roundof16total=roundof16total[['Home','Away','Results']]
        result = {0:'Home', 1:'Away'}
        roundof16total['Results'] = roundof16total['Results'].map(result)

        st.write(roundof16total)

        roundof16total['quarterfinal'] = 0
        roundof16total['quarterfinal'][roundof16total['Results']=='Home']=roundof16total.loc[:,'Home'].values
        roundof16total['quarterfinal'][roundof16total['Results']=='Away']=roundof16total.loc[:,'Away'].values

        quarterfinal = roundof16total[['quarterfinal']]

        second = quarterfinal.iloc[1::2].reset_index().iloc[:,1:]
        first = quarterfinal.iloc[::2].reset_index().iloc[:,1:]

        st.subheader(':red[QuarterFinals]:soccer:')

        quarterfinal = pd.concat([first,second],axis=1)
        quarterfinal.set_axis(['Home','Away'], axis=1, inplace=True)

        table1=pd.merge(quarterfinal, HomeStats,'left', left_on='Home',right_on='HomeClubName')
        quarterfinal=pd.merge(table1, AwayStats,'left',left_on='Away',right_on='AwayClubName')

        quarterfinal=quarterfinal.drop(['HomeClubName','AwayClubName'],axis=1)

        ## Standard Scaler
        scaler.fit(quarterfinal.iloc[:,2:])
        scaled_feat1=scaler.transform(quarterfinal.iloc[:,2:])
        Xclquarters=pd.DataFrame(scaled_feat1,columns = list(quarterfinal.iloc[:,2:].columns))

        predcl=model_LR.predict(Xclquarters)

        quarterfinal['Results']=predcl
        quarterfinal=quarterfinal[['Home','Away','Results']]
        quarterfinal['Results'] = quarterfinal['Results'].map(result)
        st.write(quarterfinal)

        quarterfinal['semifinal'] = 0
        quarterfinal['semifinal'][quarterfinal['Results']=='Home']=quarterfinal.loc[:,'Home'].values
        quarterfinal['semifinal'][quarterfinal['Results']=='Away']=quarterfinal.loc[:,'Away'].values

        semifinal = quarterfinal[['semifinal']]

        second = semifinal.iloc[1::2].reset_index().iloc[:,1:]
        first = semifinal.iloc[::2].reset_index().iloc[:,1:]

        st.subheader(':red[Semifinals]:soccer:')

        semifinal = pd.concat([first,second],axis=1)
        semifinal.set_axis(['Home','Away'], axis=1, inplace=True)

        table1=pd.merge(semifinal, HomeStats,'left', left_on='Home',right_on='HomeClubName')
        semifinal=pd.merge(table1, AwayStats,'left',left_on='Away',right_on='AwayClubName')

        semifinal=semifinal.drop(['HomeClubName','AwayClubName'],axis=1)

        ## Standard Scaler
        scaler.fit(semifinal.iloc[:,2:])
        scaled_feat1=scaler.transform(semifinal.iloc[:,2:])
        Xclsemi=pd.DataFrame(scaled_feat1,columns = list(semifinal.iloc[:,2:].columns))
        predcl=model_LR.predict(Xclsemi)

        semifinal['Results']=predcl
        semifinal=semifinal[['Home','Away','Results']]
        semifinal['Results'] = semifinal['Results'].map(result)
        st.write(semifinal)

        semifinal['final'] = 0
        semifinal['final'][semifinal['Results']=='Home']=semifinal.loc[:,'Home'].values
        semifinal['final'][semifinal['Results']=='Away']=semifinal.loc[:,'Away'].values

        semifinal = semifinal[['final']]

        second = semifinal.iloc[1::2].reset_index().iloc[:,1:]
        first = semifinal.iloc[::2].reset_index().iloc[:,1:]

        st.subheader(':red[Final]:soccer:')

        final = pd.concat([first,second],axis=1)
        final.set_axis(['Home','Away'], axis=1, inplace=True)

        table1=pd.merge(final, HomeStats,'left', left_on='Home',right_on='HomeClubName')
        final=pd.merge(table1, AwayStats,'left',left_on='Away',right_on='AwayClubName')

        final=final.drop(['HomeClubName','AwayClubName'],axis=1)

        ## Standard Scaler
        scaler.fit(final.iloc[:,2:])
        scaled_feat1=scaler.transform(final.iloc[:,2:])
        scaled_feat1 = scaler.fit_transform(final.iloc[:,2:])
        Xclfinal=pd.DataFrame(scaled_feat1,columns = list(final.iloc[:,2:].columns))
        #Xclfinal = final.iloc[:,2:]          

        predcl=model_LR.predict(Xclfinal)

        final['Results']=predcl
        final=final[['Home','Away','Results']]
        final['Results'] = final['Results'].map(result)
        st.write(final)

except:
     print("")

#Finally, in the Terminal, I have to write: python -m streamlit run PoisonousMushroom_app.py
