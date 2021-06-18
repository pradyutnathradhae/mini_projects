from numpy.lib.shape_base import tile
import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

################################################################

#Interface_Code

st.set_page_config(page_title="AUTOML APP",layout='wide')

st.write("""
         AutoMl App Designing with Streamlit library and Random Forest Regression
        
        The Following Code shows the HyperParameters and plots a 3-D Contour.
        
        CODED By:  Mr. Pradyut Nath,MSIT.""")

################## Sidebar
#Uploading File Part
st.sidebar.header("Upload Your CSV File")
uploaded_data = st.sidebar.file_uploader("[ Upload Here! ]",type=["csv"],accept_multiple_files=False)
st.sidebar.markdown("""[ Example CSV File! @@@ ](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv))""")

#Setting Parameters
st.sidebar.header("Set Training Parameters")
split_percent = st.sidebar.slider("Data split ratio (\% for Training Set)",10,90,90,5)

#Learning Parameters for RF
st.sidebar.subheader("Learning Parameters")
n_estimators = st.sidebar.slider("Number of Estimators in Random Forest (n_estimators)",0,500,(10,50),50)

stepsize_nestimators = st.sidebar.number_input("Step Size of n_estimators",10)
st.sidebar.write("--------------------------------")

max_features = st.sidebar.slider("Max Features",1,50,(1,3),1)

stepsize_maxfeatures = st.sidebar.number_input("Step Size of Max Features",1)
st.sidebar.write("--------------------------------")

min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
min_samples_leaf = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

#General Parameters for RF
st.sidebar.subheader("General Parameters")
random_state = st.sidebar.slider('Seed number (random_state)',0,1000,40,1)
criterion = st.sidebar.select_slider('Performance measure (criterion)',options=['mse','mae'])
bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)',options=[True,False])
oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the ($R^2$) on unseen data (oob_score)',options=[False,True])
n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)',options=[1,-1])

#Creating a aggregated Range values for n_estimators and max_features to be used in GridSearchCV() for Hyperparamater tuning
n_estimators_range = np.arange(n_estimators[0],n_estimators[1]+stepsize_nestimators,stepsize_nestimators)
max_features_range = np.arange(max_features[0],max_features[1]+stepsize_maxfeatures,stepsize_maxfeatures)
param_grid = dict()
param_grid['max_features'] = max_features_range
param_grid['n_estimators'] = n_estimators_range

################## Main Panel Designing

st.subheader('DATASET:')

#Model Building

def file_download(df):
    csv_file = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv_file.encode()).decode() # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64_csv}" download = "Model_Evaluation.csv" >Download CSV!</a>'
    return href

def build_model(df):
    col_names = list(df.columns)
    # col_name = st.select_slider('Select the column to be taken as "Y" ',options=col_names)
    col_name = st.radio('Select the column to be taken as "Y" ', options=col_names)
    if st.button('Submit!'):
    # Y_index = df.columns.get_loc(col_name)
        col_names.remove(col_name)
        X = df.loc[:,col_names]  #Forming X and Y
        Y = df.loc[:,[col_name]]
        
        st.markdown('A model is being built to predict the following **Y** variable:')
        st.info(col_name)
        
        #Splitting Data into train & test 
        train_x,test_x,train_y,test_y = train_test_split(X,Y,train_size=split_percent)
        train_y = np.ravel(train_y)
        test_y = np.ravel(test_y)
        rf = RandomForestRegressor(n_estimators=n_estimators,criterion=criterion,
                                min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
                                max_features=max_features,bootstrap = bootstrap,
                                oob_score=oob_score,n_jobs=n_jobs,random_state=random_state)
        
        grid = GridSearchCV(rf,param_grid=param_grid,cv=5)
        grid.fit(train_x,train_y)
        
        st.subheader('Model Performance:')
        
        y_pred = grid.predict(test_x)
        st.write('Coefficient of determination ($R^2$):')
        st.info(r2_score(test_y,y_pred))
        
        err = 'MSE' if criterion == 'mse' else 'MAE'
        err_val = mean_squared_error(test_y,y_pred) if criterion == 'mse' else mean_absolute_error(test_y,y_pred)
        st.write('{} Error:'.format(err))
        st.info(err_val)
        
        st.write('The best parameters are {} with best score = {}.'.format(grid.best_params_,grid.best_score_))
        
        #Model Parameters
        st.subheader('Model Parameters')
        st.write(grid.get_params())
        
        #Process Grid Data
        grid_results = pd.concat([pd.DataFrame(grid.cv_results_['params']),pd.DataFrame(grid.cv_results_['mean_test_score'],columns=['R2'])],axis=1)
        # Group data into groups based on the 2 hyperparameters
        grid_temp = grid_results.groupby(['max_features','n_estimators']).mean()
        # Pivoting the data:technique that rotates data from a state of rows to a state of columns, possibly aggregating 
        # multiple source values into the same target row and column intersection.
        grid_reset = grid_temp.reset_index()
        grid_reset.columns = ['max_features', 'n_estimators', 'R2']
        grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
        
        #Initializing x,y,z for plotting
        x = grid_pivot.columns.levels[1].values
        y = grid_pivot.index.values
        z = grid_pivot.values
        
        #Plot a 3D using plotly
        layout = go.Layout(
                    xaxis = go.layout.XAxis(
                        title = go.layout.xaxis.Title(
                            text = 'n_estimators'
                        )),
                    yaxis = go.layout.YAxis(
                        title = go.layout.yaxis.Title(
                            text = 'max_features'
                        )
                    )
                    )
        fig = go.Figure(data= [go.Surface(x=x,y=y,z=z)],layout=layout)
        fig.update_layout(
            title = 'HyperTuning Parameters',
            scene = dict(xaxis_title = 'n_estimators',
            yaxis_title = 'max_features',
            zaxis_title = 'R2'),
            autosize=False,
            width=900, height=900,
            margin=dict(l=65, r=50, b=65, t=90)
        )
        st.plotly_chart(fig)
        
        #Save Grid Data
        df = pd.concat([pd.DataFrame(x),pd.DataFrame(y),pd.DataFrame(z)],axis=1)
        st.markdown(file_download(df),unsafe_allow_html=True)
    
if uploaded_data is not None:
    df = pd.read_csv(uploaded_data)
    st.write(df)
    build_model(df)
else:
    st.info("Awaiting For Dataset to be uploaded:")
    if st.button("Use Example Dataset"):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name='response')
        df = pd.concat( [X,Y], axis=1 )
        
        st.markdown('The Default Diabetes dataset is used as the example.')
        st.write(df)

        build_model(df) 
