#Comments
#Author:Yogesh changes:initial file date:4/6/2018

"""
how to run this code?

1.Go to cmd prompt:
2.type python analytics4-h2o-1-fidelity-glm.py <<arguments>>
  The following are the arguments:
     1.<train_csv_file>
     2.<test_csv_file>
     .<target variable>
     4.<temp folder to contain the model_file,description_file,model_binary>
     5.<artifacts folder to contain the performance file>
     6.<AnalyticsModelFileNameSuffix>
     7.<AnalyticsModelArtifactsFileNameSuffix>

     Hyperparameters:
     8.seed
     9.family
     10.solver
     11.alpha
     12.lambda
     .iterations
     14.nfolds
     15.lambda_search

     
     16.<modelVersionNumber>
     17.<modelName>
     18.<projectName>

Sample Run
D:/Apps/fidelity_413_train.csv D:/Apps/fidelity_413_test.csv alternative_change_failure_c_flag C
:/Users/yogesh/Desktop/Yogi/h2o/temp C:/Users/yogesh/Desktop/Yogi/h2o/artifacts
LR Logit 1244 binomial AUTO 0.5 0.00 1000 100 True True v1 LogisticRegressionFidelity project1

"""
#imports

#Data manipulation libraries
import pandas as pd
import copy

#Data Serialization libraries
import pickle
from operator import itemgetter
#processing command line arguments
import sys

#source imports
from sqlalchemy import create_engine #for postgre db
import urllib #reading urls
import xml.etree.ElementTree #reading xml
import json

#numeric computation libraries
import math
import numpy as np


#visualization library
import matplotlib.pyplot as plt
import seaborn as sbn

#model libraries
from sklearn.linear_model import LogisticRegression
import h2o
from h2o.estimators import H2OGeneralizedLinearEstimator

#feature Engineering libraries
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

#metrics libraries
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

#preprocessing libraries
from sklearn.preprocessing import LabelEncoder
#from sklearn.cross_validation import train_test_split
debug=False

h2o.init(nthreads=-1,strict_version_check=True)


class source:
    
     __s_parameter_dict={'csv':{'parameters':{'filename':None}},'db':{'parameters':{'vendor':None,'driver':None,'host':None,'port':None,'username':None,'password':None,'database':None,'table':None}},'tsv':{'parameters':{'filename':None}},'json':{'parameters':{'filename':None}},'url':{'parameters':{'link':None}},'xml':{'parameters':{'filename':None}}}
	
     def __init__(self,source_type):
 
         self.source_type=source_type
	    
         self.parameters=copy.deepcopy(self.__s_parameter_dict[self.source_type]['parameters'])

     def update_parameters_to_source(self,param,val):
         
         self.parameters[param]=val
	
     def read_source(self):
         
         if(self.source_type=='csv'):
             input_df=pd.read_csv(self.parameters['filename'])
             return input_df
         if(self.source_type=='db'):
             if(self.parameters['vendor']=='postgre'):
                 engine_string="postgresql://"+str(self.parameters['username'])+":"+str(self.parameters['password'])+"@"+str(self.parameters['host'])+":"+str(int(self.parameters['port'] or 0))+"/"+str(self.parameters['database'])
                     #if debug:
                 print("Engine String:"+engine_string)
                 engine=create_engine(engine_string)
                 input_df=pd.read_sql_query("SELECT * FROM "+self.parameters['table'],engine)
                 return input_df
             if(self.parameters['vendor']=='Mysql'):
                 pass
             if(self.parameters['vendor']=='Mssql'):
                 #code for Mssql
                 pass
         if(self.source_type=='tsv'):
             input_df=pd.read_table(self.parameters['filename'])
             return input_df
         if(self.source_type=='json'):
             pass
         if(self.source_type=='url'):
             pass
         if(self.source_type=='xml'):
             pass


class model:
     
     __model_dict_c={1:'Logistic Regression',2:'Decision Trees Classifier',3:'NaiveBayesClassifier',4:'Support Vector Machines Classifier',5:'Random Forest Classifier',6:'Gradient Boosted Trees Classifier',7:'ExtraTreesClassifier',8:'Stochastic Gradient Descent Classifier'}
     
     __model_dict_r={1:'Linear Regression',2:'Decision Tree Regression',8:'Bayesian Ridge Regression',4:'Support Vector Machines Regressor',5:'Random Forest Regressor',6:'Gradient Boosted Trees Regressor',7:'ExtraTreesRegressor',8:'Stochastic Gradient Descent Regression'}
	 
     __m_parameter_dict_c={1:{'name':'Logistic Regression','parameters':{'penalty':'l2','dual':False,'tol':1e-4,'C':1.0,'fit_intercept':True,'intercept_scaling':1,'class_weight':'balanced','random_state':None,'solver':'liblinear','max_iter':100,'multi_class':'ovr','verbose':0,'warm_start':False,'n_jobs':-1}},2:{'name':'Decision Trees Classifier','parameters':{'criterion':'gini','splitter':'best','max_depth':None,'min_samples_split':2,'min_samples_leaf':1,'min_weight_fraction_leaf':0,'max_features':None,'random_state':None,'max_leaf_nodes':None,'min_impurity_decrease':0,'class_weight':None,'presort':False}},3:{'name':'NaiveBayesClassifier','parameters':{'priors':None}},4:{'name':'Support Vector Machines Classifier','parameters':{'C':1.0,'kernel':'rbf','degree':3,'gamma':'auto','coef0':0.0,'probability':False,'shrinking':True,'tol':1e-3,'cache_size':500,'class_weight':'balanced','verbose':False,'max_iter':-1,'decision_function_shape':'ovr','random_state':None}},5:{'name':'Random Forest Classifier','parameters':{'n_estimators':250,'criterion':'gini','max_features':'auto','max_depth':None,'min_samples_split':2,'min_samples_leaf':1,'min_weight_fraction_leaf':0,'max_leaf_nodes':None,'min_impurity_decrease':0,'boostrap':True,'oob_score':False,'n_jobs':1,'random_state':None,'verbose':0,'warm_start':False,'class_weight':'balanced'}},6:{'name':'Gradient Boosted Trees Classifier','parameters':{'loss':'deviance','learning_rate':0.1,'n_estimators':250,'max_depth':3,'criterion':'fried_mse','min_samples_split':2,'min_samples_leaf':1,'min_weight_fraction_leaf':0,'subsample':1.0,'max_features':None,'max_leaf_nodes':None,'min_impurity_decrease':0,'init':None,'verbose':0,'warm_start':False,'random_state':None,'presort':'auto'}},7:{'name':'ExtraTreesClassifier','parameters':{'n_estimators':250,'criterion':'gini','max_features':'auto','max_depth':None,'min_samples_split':2,'min_samples_leaf':1,'min_weight_fraction_leaf':0,'max_leaf_nodes':None,'min_impurity_decrease':0,'bootstrap':False,'oob_score':False,'n_jobs':-1,'random_state':None,'verbose':0,'warm_start':False,'class_weight':'balanced'}},8:{'name':'Stochastic Gradient Descent Classifier','parameters':{'loss':'squared_loss', 'penalty':'l2', 'alpha':0.0001, 'l1_ratio':0.15, 'fit_intercept':True, 'max_iter':None, 'tol':None, 'shuffle':True, 'verbose':0, 'epsilon':0.1, 'n_jobs':1,'random_state':None, 'learning_rate':'optimal', 'eta0':0.01, 'power_t':0.25, 'class_weight':None, 'warm_start':False, 'average':False, 'n_iter':None}}}
	 
     __m_parameter_dict_r={1:{'name':'Linear Regression','parameters':{'fit_intercept':True,'normalize':False,'copy_X':True,'n_jobs':-1}},2:{'name':'Decision Tree Regression','parameters':{'criterion':'mse', 'splitter':'best', 'max_depth':None,'min_samples_split':2,'min_samples_leaf':1,'min_weight_fraction_leaf':0.0,'max_features':None, 'random_state':None, 'max_leaf_nodes':None, 'min_impurity_decrease':0.0, 'min_impurity_split':None, 'presort':False}},3:{'name':'Bayesian Ridge Regressor','parameters':{'n_iter':300, 'tol':0.001, 'alpha_1':1e-06, 'alpha_2':1e-06, 'lambda_1':1e-06, 'lambda_2':1e-06, 'compute_score':False, 'fit_intercept':True, 'normalize':False, 'copy_X':True, 'verbose':False}},4:{'name':'Support Vector machines Regressor','parameters':{'kernel':'rbf', 'degree':3, 'gamma':'auto', 'coef0':0.0, 'tol':0.001, 'C':1.0, 'epsilon':0.1, 'shrinking':True, 'cache_size':200, 'verbose':False, 'max_iter':-1}},5:{'name':'Random Forest Regressor','parameters':{'n_estimators':10, 'criterion':'mse', 'max_depth':None, 'min_samples_split':1, 'min_samples_leaf':1, 'min_density':0.1, 'max_features':'auto', 'bootstrap':True, 'compute_importances':False, 'oob_score':False, 'n_jobs':1, 'random_state':None, 'verbose':0}},6:{'name':'Gradient Boosted Trees Regressor','parameters':{'loss':'ls', 'learning_rate':0.1, 'n_estimators':250, 'subsample':1.0, 'criterion':'friedman_mse', 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_depth':3, 'min_impurity_decrease':0.0, 'min_impurity_split':None, 'init':None, 'random_state':None, 'max_features':None, 'alpha':0.9, 'verbose':0, 'max_leaf_nodes':None, 'warm_start':False, 'presort':'auto'}},7:{'name':'ExtraTreesRegressor','parameters':{'n_estimators':250, 'criterion':'mse', 'max_depth':None, 'min_samples_split':2, 'min_samples_leaf':1, 'min_weight_fraction_leaf':0.0, 'max_features':'auto', 'max_leaf_nodes':None, 'min_impurity_decrease':0.0, 'min_impurity_split':None, 'bootstrap':False, 'oob_score':False, 'n_jobs':1, 'random_state':None, 'verbose':0, 'warm_start':False}},8:{'name':'Stochastic Gradient Descent Regression','parameters':{'loss':'squared_loss', 'penalty':'l2', 'alpha':0.0001, 'l1_ratio':0.15, 'fit_intercept':True, 'max_iter':None, 'tol':None, 'shuffle':True, 'verbose':0, 'epsilon':0.1, 'random_state':None, 'learning_rate':'invscaling', 'eta0':0.01, 'power_t':0.25, 'warm_start':False, 'average':False, 'n_iter':None}}}

     __m_parameter_dict_c_h2o={1:{'name':'Logistic Regression','parameters':{'seed':1244,'family':'binomial','solver':'AUTO','alpha':0.5,'lambda_':0.0,'max_iterations':1000,'nfolds':100,'remove_collinear_columns':True,'compute_p_values':True,'interactions':[]}}}
     #__model_simple_dict=
     #__m_parameter_dict_c_h2o={1:{'name':'Logistic Regression','parameters':{'seed':1244,'family':'binomial','solver':'AUTO','alpha':0.5,'lambda_':0.0,'max_iterations':100,'nfolds':10}}}
     
	 
    
     def __init__(self,model_type,index):
		 
         self.model_type=model_type
         self.index=index
         if(self.model_type=='C'):
             self.parameters=copy.deepcopy(self.__m_parameter_dict_c_h2o[index]['parameters'])
         else:
             self.parameters=copy.deepcopy(self.__m_parameter_dict_r[index]['parameters'])
             
         self.p_model=self.construct_model()		
       
     def add_new_model_to_model_dict(self,model_type,index,params):
         pass
         '''
         self.model
         if(index in self.__model_dict or  in self.__model_dict.values()):
             print("please dont change the contents of values already present. the id or name is already present")
         else:
             self.__model_dict[index]=name
             self.__m_parameter_dict[index]['parameters']=params
         '''       
     def update_parameter_to_model(self,param,val):

         if(type(self.parameters[param])is int):
              val=int(val)
         if(type(self.parameters[param]) is float):
              val=float(val)
         if(val=="False"):
              val=False
         if(val=="True"):
              val=True
         if(val=="None"):
              val=None
            
         self.parameters[param]=val
         self.p_model=self.construct_model()
     
     def construct_model(self):
         
         if(self.model_type=='C'):
             if(self.index==1):
                 p_model=H2OGeneralizedLinearEstimator(**self.parameters)
             if(self.index==2):
                 p_model=DecisionTreeClassifier(**self.parameters)
             if(self.index==3):
                 p_model=GaussianNB(**self.parameters)
             if(self.index==4):
                 p_model=SVC(**self.parameters)
             if(self.index==5):
                 p_model=RandomForestClassifier(**self.parameters)
             if(self.index==6):
                 p_model=GradientBoostingClassifier(**self.paraemters)
             if(self.index==7):
                 p_model=ExtraTreesClassifier(**self.parameters)
             if(self.index==8):
                 p_model=SGDClassifier(**self.parameters)
         else:
             if(self.index==1):
                 p_model=LinearRegression(**self.parameters)
             if(self.index==2):
                 p_model=DecisionTreeClassifier(**self.parameters)
             if(self.index==3):
                 p_model=BayesianRidge(**self.parameters)
             if(self.index==4):
                 p_model=SVR(**self.parameters)
             if(self.index==5):
                 p_model=RandomForestRegressor(**self.parameters)
             if(self.index==6):
                 p_model=GradientBoostingRegressor(**self.parameters)
             if(self.index==7):
                 p_model=ExtraTreesRegressor(**self.parameters)			 
             if(self.index==8):
                 p_model=SGDRegressor(**self.parameters)
         return p_model		 
         
		

class Analytics:
     
     def __init__(self):
    	 
         self.sources=[]
         self.models=[]
         self.train_source=None
         self.test_source=None
         self.Scoring_source=None
         self.model1=None
         self.model2=None
         self.model1_pred_prob=None
         self.model2_pred_prob=None
         
     def readme(self):
    
         print ("Machine learning. We at analytics team perform modelling of data to produce insights and predictions.We do classification,regression,Clustering, currently. We plan to do Natural Language Processing and Recommendation Systems")

     def help(self,choice):
         
         if(choice=='C'):
             print("The following are the Available Classification models and the ids for the models")
             print ("1.Logistic Regression")
             print ("2.Decision Trees Classification")
             print ("3.NaiveBayesClassifier")
             print ("4.Support Vector Machines")
             print ("5.Random Forest Classifier")
             print ("6.Gradient Boosted Trees Classifier") 
             print ("7.ExtraTreesClassifier")
             print ("8.Stochastic Gradient Descent Classifier")
         elif(choice=='R'):
             print ("The following are the Avaiable Regression Models and the ids for the models")
             print ("1.Linear Regression")
             print ("2.Decision Tree Regression")
             print ("3.Bayesian Ridge Regression")
             print ("4.Support Vector Machines Regressor")
             print ("5.Random Forest Regressor")
             print ("6.Gradient Boosted Trees Regressor")
             print ("7.ExtraTrees Regressor")
             print ("8.Stochastic Gradient Descent Regressor")
         if(choice=='S'): 
             print ("The following are the types of input source formats we take")
             print ("1.csv:comma separated values file")
             print ("2.tsv:tab separated values file")
             print ("3.db:database( Mysql,Mssql,Postgre,ORACLE)")
             print ("4.json:json file")
             print ("5.url:url link")
             print ("6.xml:xml file")

     def one_by(self,x):
          if(x==0):
               return 1
          else:
               return 1/(x)

     def log_of(self,x):
          if(x<=0):
               return 0
          else:
               return math.log(x)

     def tanh(self,x):
          return math.tanh(x)

     def sqrt(self,x):
          if(x<0):
               x=-x
               return 0-math.sqrt(x)
          else:
               return math.sqrt(x)
          

     def initialize(self):
         for j in range(len(sys.argv)):
            print(str(j)+":"+sys.argv[j])
         self.train_source=source("csv")
         self.train_source.parameters['filename']=sys.argv[1]
         source_dataframe=self.train_source.read_source()
         source_dataframe['assignee_prior_changes_c']=(source_dataframe['assignee_prior_changes_c']-source_dataframe['assignee_prior_changes_c'].mean())/source_dataframe['assignee_prior_changes_c'].std()
         source_dataframe['one_by_apfr']=source_dataframe['assignee_prior_failure_rate'].apply(self.one_by)
         source_dataframe['log_of_apfr']=source_dataframe['assignee_prior_failure_rate'].apply(self.log_of)
         source_dataframe['mult_apfr_apc']=source_dataframe.assignee_prior_failure_rate*source_dataframe.assignee_prior_changes_c
         source_dataframe['csq_apfr_apc']=source_dataframe.assignee_prior_failure_rate*source_dataframe.assignee_prior_failure_rate+source_dataframe.assignee_prior_changes_c*source_dataframe.assignee_prior_changes_c
         source_dataframe['add_apfr_apc']=source_dataframe.assignee_prior_failure_rate+source_dataframe.assignee_prior_changes_c
         source_dataframe['tanh_apc']=source_dataframe['assignee_prior_changes_c'].apply(self.tanh)
         source_dataframe['tanh_apfr']=source_dataframe['assignee_prior_failure_rate'].apply(self.tanh)
         source_dataframe['sqrt_apc']=source_dataframe['assignee_prior_changes_c'].apply(self.sqrt)
         source_dataframe['sqrt_apfr']=source_dataframe['assignee_prior_failure_rate'].apply(self.sqrt)
         source_dataframe['cube_apfr']=source_dataframe.assignee_prior_failure_rate*source_dataframe.assignee_prior_failure_rate*source_dataframe.assignee_prior_failure_rate
         source_dataframe['cube_apc']=source_dataframe.assignee_prior_changes_c*source_dataframe.assignee_prior_changes_c*source_dataframe.assignee_prior_changes_c
         
         source_dataframe.to_csv(sys.argv[4]+"/Logistic_Regression_source_dataframe.csv",encoding='utf-8')
         #source_dataframe_h2o=h2o.H2OFrame(source_dataframe)

         #normalize the numerical columns
         #source_dataframe['one_by_apfr']=(source_dataframe['one_by_apfr']-source_dataframe['one_by_apfr'].mean())/source_dataframe['one_by_apfr'].std()
         #source_dataframe['log_of_apfr']=(source_dataframe['log_of_apfr']-source_dataframe['log_of_apfr'].mean())/source_dataframe['log_of_apfr'].std()
         #source_dataframe['mult_apfr_apc']
         #train['one_by_apfr']=(train['one_by_apfr']-train['one_by_apfr'].mean())/train['one_by_apfr'].std()
         
         self.model1=model('C',1)

         y=8
         for x in self.model1.parameters:
              if x == 'interactions':
                   pass
              else:
                   self.model1.update_parameter_to_model(x,sys.argv[y])
              
              y=y+1
         interaction_list=['assignee_prior_changes_c','assignee_prior_failure_rate']
         self.model1.update_parameter_to_model("interactions",interaction_list)
         print(self.model1.parameters)                   
         self.train_data(source_dataframe,sys.argv[3])
         
     def collect_all_data(self):
         #concat can lead to memory errors in python
         df=pd.DataFrame()
         for each_source in sources:
             df1=each_source.read_source()
             df=pd.concat(df,df1)
         return df
     	
     def clean_data(self,df):
         #1.NORMALIZING COLUMNS
         #df['column_name']=(df['column_name']-df['column_name'].mean())/df['column_name'].std()
		 
         #2.COLUMN SELECTION
         #df=pcsv[['benefit_organization_flag','assignee_total_changes','assignee_failure_rate' ,'as_total_changes','ag_failure_rate','new_copy_method','outside_maintenance_schedule','type','alternate_programmer_flag','impact_on_continuity_plan','audit_violation_verified','odate_flag','new_copy','tier_4_restricted_access_required','impact_on_capacity_plan','programmer_flag','cio','incident_flag','is_application_outage_required','bu_cab','manually_added_groups_flag','u_ecm','impact_on_availability_plan','ppmc_request_flag','are_fsc_services_required','template_flag','risk','template_type','total_opdocs_flag','programmer_work_number_flag','parent_flag','is_server_outage_required','after_hours_contact_flag','impact_if_not_implemented_flag','conflict_status','category','business_units_affected_flag','environment_details','bu_oversight_group_flag' ]]
		 
         #.DIAGNOSE THE LABEL DISTRIBUTION
         #print(pd.crosstab(df['column1'],df['column2']))
		 
         #4.NULL VALUES
         #4.1 Removing Null columns/rows
         #  1.removing columns with null values
         #df=df.dropna(subset=['Age'])
		 
         #  2.removing column with all null values
         #df=df.dropna(axis=1,how='all')
		 
         #  .removing rows with all null values
         #df=df.dropna(axis=0,how='all')
		 
         #  4.removing columns with atleast 2 null values
         #df=df.dropna(axis=1,thresh=2)
		 
         #4.2 Null value imputation
         #4.2.1 numerical variables with mean
         #mean=df['age'].mean() 
         #df['age']=df['age'].fillna(mean)
         #problem with mean imputation is that it is heavily affected by outliers. go with median.
         #median=df['age'].median()
         #df['age']=df['age'].fillna(median)
		 
         #4. Imputing variables with mice
         # what is mice? mice is the iterative idea of applying ml algorithms to repeatedly impute missing data for all the columns in a dataset.it starts with mean imputation for a column with missing values.then it does the same for all the columns with missing data.then on the next iteration it computes linear regression with dependent variable as the column with missing vaue and the independent variables as all other variables.MICE procedures assume that the data are Missing At Random.
         #df = data.drop(['Survived'], axis=1)
         #column_titles = list(df)
         #mice_results = fancyimpute.MICE().complete(np.array(df)) .Fancy impute requires microsoft visual studio c++ compiler.
         #results = pd.DataFrame(mice_results, columns=column_titles)
		  
         #5.MIN MAX SCALING
         #np_minmax = (x_np - x_np.min()) / (x_np.max() - x_np.min())
         #MinMaxScaler is very sensitive to the presence of outliers.
		 
         #6.OUTLIERS
         #1.simple way is to find the observation which is 2-3 standard deviation away from the mean.But this method is affected by outliers.
         #2.Isolation forest is a way to go when ml needs to be used to detect them.
         #3.Multivariate gaussian function can be used and the probability of occurence of an observation is beyond a threshold, then it is an outlier.
         #4.using mean absolute deviation.
         # threshold = 3.5
         # median_y = np.median(ys)
         # median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
         # modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ys]
         # return np.where(np.abs(modified_z_scores) > threshold)
		 
         #7.CATEGORY TO INTEGER CONVERSION
         # pcsv2=pd.DataFrame({col: pcsv1[col].astype('category').cat.codes for col in pcsv1}, index=pcsv1.index)

         #8.CONTINUOUS VARIABLE TO RANGES
         # qcut:Discretize variable into equal-sized buckets based on rank or based on sample quantiles. For example 1000 values for 10 quantiles would produce a Categorical object indicating quantile membership for each data point.
         # array=pd.cut(df['column'],number_of_ranges) for division based on column values.
         # array=pd.qcut(df['coulmn'],number_of_bins) for evenly distributed bins.

		 
		 
         return df

     def train_data(self,df,target_column):
         
         self.test_source=source("csv")
         self.test_source.parameters['filename']=sys.argv[2]
         test=self.test_source.read_source()
         test['assignee_prior_changes_c']=(test['assignee_prior_changes_c']-test['assignee_prior_changes_c'].mean())/test['assignee_prior_changes_c'].std()
         
         test['one_by_apfr']=test['assignee_prior_failure_rate'].apply(self.one_by)
         test['log_of_apfr']=test['assignee_prior_failure_rate'].apply(self.log_of)
         test['mult_apfr_apc']=test.assignee_prior_failure_rate*test.assignee_prior_changes_c
         test['csq_apfr_apc']=test.assignee_prior_failure_rate*test.assignee_prior_failure_rate+test.assignee_prior_changes_c*test.assignee_prior_changes_c
         test['add_apfr_apc']=test.assignee_prior_failure_rate+test.assignee_prior_changes_c
         test['tanh_apc']=test['assignee_prior_changes_c'].apply(self.tanh)
         test['tanh_apfr']=test['assignee_prior_failure_rate'].apply(self.tanh)
         test['sqrt_apc']=test['assignee_prior_changes_c'].apply(self.sqrt)
         test['sqrt_apfr']=test['assignee_prior_failure_rate'].apply(self.sqrt)
         test['cube_apfr']=test.assignee_prior_failure_rate*test.assignee_prior_failure_rate*test.assignee_prior_failure_rate
         test['cube_apc']=test.assignee_prior_changes_c*test.assignee_prior_changes_c*test.assignee_prior_changes_c
         
         #normalize the numerical columns
         
         test_h2o=h2o.H2OFrame(test)
         target=target_column

         predictors=[x for x in df if x not in [target]]
         source_dataframe_h2o=h2o.H2OFrame(df)
         #print(predictors)
         
         self.model1.p_model.train(x=predictors,y=target,training_frame=source_dataframe_h2o)
         print(self.model1.p_model.coef())
         model_coefficients=self.model1.p_model.coef()
         
		 model_p_values=self.model1.p_model._model_json['output']['coefficients_table'].as_data_frame()
         #model_p_values=model_p_values[model_p_values['coefficients']>0]
         #model_p_values=model_p_values[model_p_values['p_value']<0.5]
         model_p_values.to_csv(sys.argv[4]+"/Logistic_Regression_p_values.csv",encoding='utf-8')
		 
         #coefficients_df=pd.DataFrame(data=self.model1.p_model.coef_[0][:],index=range(len(self.model1.p_model.coef_[0])),columns=['coefficients'])
         #predictors_df=pd.DataFrame(data=predictors,index=range(len(predictors)),columns=['predictors'])
         #predictors_df['predictors'][0]="Intercept"
         #coefficients_df['predictors']=predictors_df['predictors']#redictors
         #coefficients_df=coefficients_df.sort_values(by=['coefficients'],ascending=False)
         
         #model_file1=open(sys.argv[4]+"/Logistic_Regression.p","wb")
         #pickle.dump(self.model1.p_model,model_file1)
         #model_file1.close()

         model_path = h2o.save_model(model=self.model1.p_model, path=sys.argv[4], force=True)
         print(model_path)

         model_filename_file=open(sys.argv[4]+"/Logistic_Regression_model_filename.txt","w+")
         model_filename_file.write(model_path)
         model_filename_file.close()
         
         #coefficients_df.to_csv(sys.argv[4]+"/Logistic_Regression_model.txt",encoding='utf-8',sep='\t')
         model_description_file=open(sys.argv[4]+"/Logistic_Regression_model_description.csv","w+")
         model_description_file.write("\n\t\tLogistic Regression Model Description \t\t\t\n\n")
         model_description_file.write("Features,Feature values,Beta Coefficients")
         
         model_describe_file=open(sys.argv[4]+"/Logistic_Regression_model.txt","w+")
         model_describe_file.write("\n\t\tLogistic Regression Model \t\t\t\n\n")

         sorted_coefficients=sorted(model_coefficients.items(),key=itemgetter(1),reverse=True)
         
         for key,value in sorted_coefficients:
             model_describe_file.write("\n"+'{0:^45}'.format(str(key))+":"+"\t"+str(value))
             string_split=key.split('.')
             
             if(value!=0):
                 if('.' in key):
                      string_split=key.split('.')
                      model_description_file.write("\n"+str(string_split[0])+","+str(string_split[1])+","+str(value))
                 else:
                      model_description_file.write("\n"+str(key)+","+","+str(value))        
         model_describe_file.close()
         model_description_file.close()
                              
         model1_predictions=self.model1.p_model.predict(test_h2o)
         model1_predictions=model1_predictions.as_data_frame()
         test_df=test_h2o.as_data_frame()
         model1_predictions['alternative_change_failure_c_flag']=test_df['alternative_change_failure_c_flag']
         model1_predictions=model1_predictions.drop('predict',axis=1)           
         model1_predictions.to_csv(sys.argv[5]+"/Logistic_Regression_Predicted_Probability.csv",encoding='utf-8')
                  
         evaluate_file=open(sys.argv[5]+"/Logistic_Regression_Performance.txt","w+")
         model1_performance=self.model1.p_model.model_performance(test_h2o)
         model1_perf_dictionary=model1_performance.__dict__
         
         model_perf_confusion_matrix=model1_performance.confusion_matrix()
         confusion_matrix=model_perf_confusion_matrix.table.as_data_frame()
         #print(confusion_matrix)
         evaluate_file.write("\n\n\n"+'{0:^45}'.format("CONFUSION MATRIX")+"\n\n")
         evaluate_file.write("\n"+'{0:^45}'.format("True Positives")+":"+str(confusion_matrix.iloc[1,2]))
         evaluate_file.write("\n"+'{0:^45}'.format("False Positives")+":"+str(confusion_matrix.iloc[0,2]))
         evaluate_file.write("\n"+'{0:^45}'.format("True Negatives")+":"+str(confusion_matrix.iloc[0,1]))
         evaluate_file.write("\n"+'{0:^45}'.format("False Negatives")+":"+str(confusion_matrix.iloc[1,1]))
         precision=confusion_matrix.iloc[1,2]/(confusion_matrix.iloc[1,2]+confusion_matrix.iloc[0,2])
         recall=confusion_matrix.iloc[1,2]/(confusion_matrix.iloc[1,2]+confusion_matrix.iloc[1,1])
         
         evaluate_file.write("\n\n\n\n"+'{0:^45}'.format("PRECISION")+":"+str(precision))
         evaluate_file.write("\n"+'{0:^45}'.format("RECALL")+":"+str(recall))

         max_scores=model1_perf_dictionary['_metric_json']['max_criteria_and_metric_scores'].as_data_frame()
         model1_perf_dictionary['_metric_json'].pop('__meta',None)
         model1_perf_dictionary['_metric_json'].pop('model',None)
         model1_perf_dictionary['_metric_json'].pop('frame',None)
         model1_perf_dictionary['_metric_json'].pop('description',None)
         model1_perf_dictionary['_metric_json'].pop('predictions',None)
         model1_perf_dictionary['_metric_json'].pop('custom_metric_name',None)
         model1_perf_dictionary['_metric_json'].pop('custom_metric_value',None)
         model1_perf_dictionary['_metric_json'].pop('thresholds_and_metric_scores',None)
         model1_perf_dictionary['_metric_json'].pop('gains_lift_table',None)
         model1_perf_dictionary['_metric_json'].pop('max_criteria_and_metric_scores',None)


         for key in model1_perf_dictionary['_metric_json']:
              evaluate_file.write("\n"+'{0:^45}'.format(str(key))+":"+str(model1_perf_dictionary['_metric_json'][key]))
         
         
         #evaluate_file.write("\n"+'{0:^45}'.format(str(max_scores['metric'][4]))+":"+str(max_scores['value'][4]))
         #evaluate_file.write("\n"+'{0:^45}'.format(str(max_scores['metric'][5]))+":"+str(max_scores['value'][5]))
         #evaluate_file.write("\n"+'{0:^45}'.format(str(max_scores['metric'][6]))+":"+str(max_scores['value'][6]))

         #print(self.model1.p_model.model_performance(test_h2o))

         #evaluate_file.write("\n\t\t\tMODEL PERFORMANCE\t\t\t\n\n")
         #evaluate_file.write("\n\t\tLogistic Regression Model Performance\n\n")
         #evaluate_file.write(classification_report(test[target],model1_predictions))
         #evaluate_file.write("\n\nCONFUSION MATRIX\n\n")
         #tn,fp,fn,tp=confusion_matrix(test[target],model1_predictions).ravel()
         #print("tp:"+str(tp)+"fp:"+str(fp)+"fn:"+str(fn)+"tn:"+str(tn))
         #evaluate_file.write("\n\nTrue Positives:"+str(tp)+"\nFalse Positives:"+str(fp)+"\nFalse Negatives:"+str(fn)+"\nTrue Negatives:"+str(tn))
         #precision=tp/(tp+fp)
         #evaluate_file.write("\nPRECISION:"+str(precision))
         #recall=tp/(tp+fn)
         #evaluate_file.write("\nRECALL:"+str(recall))
         #evaluate_file.write("\n AUC:"+str(metrics.roc_auc_score(test[target],model1_predictions)))
         
         evaluate_file.write("\n\n Logistic Regression Hyper parameters Used\n\n")
         print("Hyper parameters Used\n\n")
         for key in N.model1.parameters:
             evaluate_file.write("\n"+'{0:^45}'.format(str(key))+":"+str(N.model1.parameters[key]))
             print(key+":"+str(N.model1.parameters[key]))
         evaluate_file.close()

         
         self.test_model(sys.argv[3])
	 
     def test_model(self,target_column):
         
         self.Scoring_source=source("csv")
         self.Scoring_source.parameters['filename']=sys.argv[2]
         Scoring_dataframe=self.Scoring_source.read_source()
         Scoring_dataframe['assignee_prior_changes_c']=(Scoring_dataframe['assignee_prior_changes_c']-Scoring_dataframe['assignee_prior_changes_c'].mean())/Scoring_dataframe['assignee_prior_changes_c'].std()
         
         Scoring_dataframe['one_by_apfr']=Scoring_dataframe['assignee_prior_failure_rate'].apply(self.one_by)
         Scoring_dataframe['log_of_apfr']=Scoring_dataframe['assignee_prior_failure_rate'].apply(self.log_of)
         Scoring_dataframe['mult_apfr_apc']=Scoring_dataframe.assignee_prior_failure_rate*Scoring_dataframe.assignee_prior_changes_c
         Scoring_dataframe['csq_apfr_apc']=Scoring_dataframe.assignee_prior_failure_rate*Scoring_dataframe.assignee_prior_failure_rate+Scoring_dataframe.assignee_prior_changes_c*Scoring_dataframe.assignee_prior_changes_c
         Scoring_dataframe['add_apfr_apc']=Scoring_dataframe.assignee_prior_failure_rate+Scoring_dataframe.assignee_prior_changes_c
         Scoring_dataframe['tanh_apc']=Scoring_dataframe['assignee_prior_changes_c'].apply(self.tanh)
         Scoring_dataframe['tanh_apfr']=Scoring_dataframe['assignee_prior_failure_rate'].apply(self.tanh)
         Scoring_dataframe['sqrt_apc']=Scoring_dataframe['assignee_prior_changes_c'].apply(self.sqrt)
         Scoring_dataframe['sqrt_apfr']=Scoring_dataframe['assignee_prior_failure_rate'].apply(self.sqrt)
         Scoring_dataframe['cube_apfr']=Scoring_dataframe.assignee_prior_failure_rate*Scoring_dataframe.assignee_prior_failure_rate*Scoring_dataframe.assignee_prior_failure_rate
         Scoring_dataframe['cube_apc']=Scoring_dataframe.assignee_prior_changes_c*Scoring_dataframe.assignee_prior_changes_c*Scoring_dataframe.assignee_prior_changes_c
         
         Scoring_h2o=h2o.H2OFrame(Scoring_dataframe)

         model1_filename_file=open(sys.argv[4]+"/Logistic_Regression_model_filename.txt","r")
         model1_filename_path=model1_filename_file.readline()
         model1_filename_file.close()

         model1_retrieved=h2o.load_model(path=str(model1_filename_path))
         #model1_file=open(sys.argv[4]+"/Logistic_Regression.p","rb")
         #model1_retrieved=pickle.load(model1_file)
         
         target=target_column

         predictors=[x for x in Scoring_dataframe if x not in [target]]
         
         model1_predictions=model1_retrieved.predict(Scoring_h2o)

         model1_predictions=model1_predictions.as_data_frame()
         model1_predictions=model1_predictions.drop('predict',axis=1)           
         model1_predictions.to_csv(sys.argv[5]+"/Logistic_Regression_Scoring_Predicted_Probability.csv",encoding='utf-8')

         model1_performance=model1_retrieved.model_performance(Scoring_h2o)
         model1_perf_dictionary=model1_performance.__dict__
         
         model_perf_confusion_matrix=model1_performance.confusion_matrix()
         confusion_matrix=model_perf_confusion_matrix.table.as_data_frame()
         #print(confusion_matrix)

         Scoring_file=open(sys.argv[5]+"/Logistic_Regression_Scoring_Performance.txt","w+")
         Scoring_file.write("\n\t\t\tSCORING PERFORMANCE\t\t\t\n\n")
         Scoring_file.write("\n\t\tLogistic Regression Model Performance\n\n")

         Scoring_file.write("\n\n\n"+'{0:^45}'.format("CONFUSION MATRIX")+"\n\n")
         Scoring_file.write("\n"+'{0:^45}'.format("True Positives")+":"+str(confusion_matrix.iloc[1,2]))
         Scoring_file.write("\n"+'{0:^45}'.format("False Positives")+":"+str(confusion_matrix.iloc[0,2]))
         Scoring_file.write("\n"+'{0:^45}'.format("True Negatives")+":"+str(confusion_matrix.iloc[0,1]))
         Scoring_file.write("\n"+'{0:^45}'.format("False Negatives")+":"+str(confusion_matrix.iloc[1,1]))
         precision=confusion_matrix.iloc[1,2]/(confusion_matrix.iloc[1,2]+confusion_matrix.iloc[0,2])
         recall=confusion_matrix.iloc[1,2]/(confusion_matrix.iloc[1,2]+confusion_matrix.iloc[1,1])
         
         Scoring_file.write("\n\n\n\n"+'{0:^45}'.format("PRECISION")+":"+str(precision))
         Scoring_file.write("\n"+'{0:^45}'.format("RECALL")+":"+str(recall))

         #max_scores=model1_perf_dictionary['_metric_json']['max_criteria_and_metric_scores'].as_data_frame()
         model1_perf_dictionary['_metric_json'].pop('__meta',None)
         model1_perf_dictionary['_metric_json'].pop('model',None)
         model1_perf_dictionary['_metric_json'].pop('frame',None)
         model1_perf_dictionary['_metric_json'].pop('description',None)
         model1_perf_dictionary['_metric_json'].pop('predictions',None)
         model1_perf_dictionary['_metric_json'].pop('custom_metric_name',None)
         model1_perf_dictionary['_metric_json'].pop('custom_metric_value',None)
         model1_perf_dictionary['_metric_json'].pop('thresholds_and_metric_scores',None)
         model1_perf_dictionary['_metric_json'].pop('gains_lift_table',None)
         model1_perf_dictionary['_metric_json'].pop('max_criteria_and_metric_scores',None)


         for key in model1_perf_dictionary['_metric_json']:
              Scoring_file.write("\n"+'{0:^45}'.format(str(key))+":"+str(model1_perf_dictionary['_metric_json'][key]))
         
         
         #pred_prob1=model1_retrieved.predict_proba(Scoring_dataframe[predictors])

         #self.model1_pred_prob=pd.DataFrame(data=pred_prob1[:,:],index=range(len(pred_prob1)),columns=['0','1'])
         #self.model1_pred_prob.to_csv(sys.argv[5]+"/Logistic_Regression_Scoring_Predicted_Probability.csv",encoding='utf-8')
         
         #Scoring_file.write(classification_report(Scoring_dataframe[target],model1_predictions))
         #Scoring_file.write("\n\nCONFUSION MATRIX\n\n")
         #tn,fp,fn,tp=confusion_matrix(Scoring_dataframe[target],model1_predictions).ravel()
         #print("tp:"+str(tp)+"fp:"+str(fp)+"fn:"+str(fn)+"tn:"+str(tn))
         #print("Logistic Regression:AUC score :%f" %(metrics.roc_auc_score(Scoring_dataframe[target],model1_predictions)))
         #Scoring_file.write("\n\nTrue Positives:"+str(tp)+"\nFalse Positives:"+str(fp)+"\nFalse Negatives:"+str(fn)+"\nTrue Negatives:"+str(tn))
         #Scoring_file.write("\nAUC score :%f" %(metrics.roc_auc_score(Scoring_dataframe[target],model1_predictions)))
         #precision=tp/(tp+fp)
         #Scoring_file.write("\nPRECISION:"+str(precision))
         #recall=tp/(tp+fn)
         #Scoring_file.write("\nRECALL:"+str(recall))
         
         #Scoring_file.write("\n\nHyper parameters Used\n\n")
         #for key in N.model1.parameters:
             #Scoring_file.write("\n"+key+":"+str(N.model1.parameters[key]))
         
         Scoring_file.close()
         
         """
         Scoring_file=open("C:/Users/yogesh/Desktop/Yogi/Decision_Scoring_Performance.txt","w+")
         Scoring_file.write("\n\t\tDecision Trees Model Performance\n\n")
         Scoring_file.write(classification_report(Scoring_dataframe[target],model2_predictions))
         Scoring_file.write("\n\nCONFUSION MATRIX\n\n")
         tn,fp,fn,tp=confusion_matrix(Scoring_dataframe[target],model2_predictions).ravel()
         print("tp:"+str(tp)+"fp:"+str(fp)+"fn:"+str(fn)+"tn:"+str(tn))
         print("Decision Trees:AUC score :%f" %(metrics.roc_auc_score(Scoring_dataframe[target],model2_predictions)))
         Scoring_file.write("\n\nTrue Positives:"+str(tp)+"\nFalse Positives:"+str(fp)+"\nFalse Negatives:"+str(fn)+"\nTrue Negatives:"+str(tn))
         Scoring_file.write("Decision Trees:AUC score :%f" %(metrics.roc_auc_score(Scoring_dataframe[target],model2_predictions)))
         precision=tp/(tp+fp)
         Scoring_file.write("\nPRECISION:"+str(precision))
         recall=tp/(tp+fn)
         Scoring_file.write("\nRECALL:"+str(recall))
         Scoring_file.close()
         """
	 
     def prepare_data(self,source_dataframe):
         pass
         """g=[row[0] for row in pred_prob[test[target]==1]]
         f=np.asarray(g)
         avg_pred_chg_failure_prob_1=f.mean()
         g1=[row[1] for row in pred_prob[test[target]==1]]
         f1=np.asarray(g1)
         avg_pred_chg_failure_prob_0=f1.mean()
	 """

if(__name__=='__main__'):
	
   N=Analytics()
   N.initialize()
   N.test_model("alt_failure_flag")


