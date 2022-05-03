import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as mt
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from scipy.interpolate import CubicSpline
from lifelines import NelsonAalenFitter
from lifelines import AalenJohansenFitter
from lmfit import Minimizer, Parameters, report_fit
from scipy.integrate import quad
from matplotlib.dates import DateFormatter

'''
Write down the objective function that we want to minimize, i.e., the residuals 
'''
def residuals_gompertz(params, t, data):
    #Get an ordered dictionary of parameter values
    v = params.valuesdict()
    #Gompertz Model
    model = v['N_max']*np.exp(-v['beta']*np.exp(-v['kappa']*t))
    #Return residuals
    return model - data
'''
Gompertz growth model
'''
def gompertz_growth(params, t):
    #Get an ordered dictionary of parameter values
    v = params
    #Gompertz Model
    model = v['N_max']*np.exp(-v['beta']*np.exp(-v['kappa']*t))
    return model 
'''
Interpolation
'''
def inter_soft(X,Y,step):
    interval=mt.floor(len(Y)/step)
    Y_inter=[]
    if len(Y)%step==0:
        Y_inter=np.empty(interval)
        X_inter=np.empty(interval)
        for i in range(interval):
            Y_inter[i]=Y[step*i]  
            X_inter[i]=X[step*i]
        if (interval-1)*step<len(Y)-1:
            Y_inter=np.append(Y_inter,Y[-1])
            X_inter=np.append(X_inter,X[-1])
        x=X_inter
        y=Y_inter
        cs_A=CubicSpline(x,y)
    else:
        Y_inter=np.empty(interval+1)
        X_inter=np.empty(interval+1)
        for i in range(interval+1):
            Y_inter[i]=Y[step*i]  
            X_inter[i]=X[step*i]
        if (interval)*step<len(Y)-1:
            Y_inter=np.append(Y_inter,Y[-1])
            X_inter=np.append(X_inter,X[-1]) 
        x=X_inter
        y=Y_inter
        cs_A=CubicSpline(x,y)
    return cs_A
'''
Gaussian Mixture
'''
def GMM(G,X):
    componentes_AIC=np.zeros(len(G))
    gm=[]
    # Evaluamos el número de olas a aplicar
    for i in range(len(G)):
        gm.append(GaussianMixture(n_components=G[i],random_state=0).fit(X))
        componentes_AIC[i]=gm[i].bic(X)
    index=np.argmin(componentes_AIC)
    return gm[index],index
'''
Obtain initial conditions for all the parameters using non parametric estimation of risk
'''
def estimate_initial_parameters_risk(G,data_sample,data_state):
    data_sample=data_sample.sample(frac=0.2)
    X=np.column_stack([data_sample['tiempo'].to_numpy()])
    # dias=np.asarray(data_state.index)
    gm=GMM(G,X)
    # Clusters
    data_sample['cluster']=gm[0].predict(X)
    data_state['cluster']=gm[0].predict(np.column_stack([data_state.index.to_numpy()]))
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    '''
    Nelson-Aalen estimation
    '''
    '''
    T=data_sample['tiempo'].to_numpy()
    E=np.ones(len(T))
    naf=NelsonAalenFitter()
    naf.fit(T,event_observed=E)
    smoothed_hazard_sample=naf.smoothed_hazard_(3.0)
    smoothed_hazard_sample['timeline']=smoothed_hazard_sample.index
    smoothed_hazard_sample=smoothed_hazard_sample.sort_values('timeline')
    timeline=smoothed_hazard_sample['timeline'].to_numpy()
    hazard=smoothed_hazard_sample['differenced-NA_estimate'].to_numpy()
    hazard_function=inter_soft(timeline,hazard,3)
    '''
    # Data and parameters per cluster
    data_clusters={}
    parameters={}
    index=0
    for i in labels:
        data_clusters[index]=data_sample.loc[data_sample['cluster']==i,['tiempo']]
        '''
        Nelson-Aalen estimation
        '''
        T=data_clusters[index]['tiempo'].to_numpy()
        E=np.ones(len(T))
        naf=NelsonAalenFitter()
        naf.fit(T,event_observed=E)
        smoothed_hazard_sample=naf.smoothed_hazard_(3.0)
        smoothed_hazard_sample['timeline']=smoothed_hazard_sample.index
        smoothed_hazard_sample=smoothed_hazard_sample.sort_values('timeline')
        timeline=smoothed_hazard_sample['timeline'].to_numpy()
        hazard=smoothed_hazard_sample['differenced-NA_estimate'].to_numpy()
        hazard_function=inter_soft(timeline,hazard,3)
        #rt=data_clusters[index]['Rt'].to_numpy()
        hazard_component=hazard_function(T)
        # Filter -inf
        dict_inf={'tiempo':T,'hazard':hazard_component}
        data_inf=pd.DataFrame(data=dict_inf)
        data_inf=data_inf[data_inf['hazard']>1e-4]
        #data_inf=data_inf[data_inf['Rt']>0.1]
        # Filtered data
        T_new=data_inf['tiempo'].to_numpy()
        rt_and_T=np.column_stack([-T_new])
        hazard_component_new=np.log(data_inf.hazard.to_numpy())
        # Linear regression
        reg=LinearRegression(fit_intercept=True).fit(rt_and_T,hazard_component_new)
        #reg_intercept=reg.coef_[0]*mean_T_new
        #reg.coef_[0]=abs(reg.coef_[0])
        parameters[index]=np.array([reg.intercept_,reg.coef_[0]])
        #parameters[index]=np.array([reg_intercept,reg.coef_[0]])
        index+=1
    parameters['prob_a_priori']=np.exp(gm[0]._estimate_log_weights())
    # Dictionary with the whole initial parameter vector
    return parameters
'''
Obtain initial conditions for all the parameters using non parametric estimation of risk with Rt
'''
def estimate_initial_parameters_risk_Rt(G,data_sample,data_state,intercept=False):
    X=np.column_stack([data_sample['tiempo'].to_numpy(),data_sample['Rt'].to_numpy()])
    # dias=np.asarray(data_state.index)
    gm=GMM(G,X)
    # Clusters
    data_sample['cluster']=gm[0].predict(X)
    data_state['cluster']=gm[0].predict(np.column_stack([data_state.index.to_numpy(),data_state.Rt.to_numpy()]))
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    '''
    Nelson-Aalen estimation
    '''
    '''
    T=data_sample['tiempo'].to_numpy()
    E=np.ones(len(T))
    naf=NelsonAalenFitter()
    naf.fit(T,event_observed=E)
    smoothed_hazard_sample=naf.smoothed_hazard_(3.0)
    smoothed_hazard_sample['timeline']=smoothed_hazard_sample.index
    smoothed_hazard_sample=smoothed_hazard_sample.sort_values('timeline')
    timeline=smoothed_hazard_sample['timeline'].to_numpy()
    hazard=smoothed_hazard_sample['differenced-NA_estimate'].to_numpy()
    hazard_function=inter_soft(timeline,hazard,3)
    '''
    # Data and parameters per cluster
    data_clusters={}
    parameters={}
    index=0
    for i in labels:
        data_clusters[index]=data_sample.loc[data_sample['cluster']==i]
        '''
        Nelson-Aalen estimation
        '''  
        T=data_clusters[index]['tiempo'].to_numpy()
        E=np.ones(len(T))
        naf=NelsonAalenFitter()
        naf.fit(T,event_observed=E)
        smoothed_hazard_sample=naf.smoothed_hazard_(3.0)
        smoothed_hazard_sample['timeline']=smoothed_hazard_sample.index
        smoothed_hazard_sample=smoothed_hazard_sample.sort_values('timeline')
        timeline=smoothed_hazard_sample['timeline'].to_numpy()
        hazard=smoothed_hazard_sample['differenced-NA_estimate'].to_numpy()
        hazard_function=inter_soft(timeline,hazard,3)
        rt=data_clusters[index]['Rt'].to_numpy()
        hazard_component=hazard_function(T)
        # Filter -inf
        dict_inf={'tiempo':T,'hazard':hazard_component,'Rt':rt}
        data_inf=pd.DataFrame(data=dict_inf)
        data_inf=data_inf[data_inf['hazard']>0.001]
        #data_inf=data_inf[data_inf['Rt']>0.001]
        # Filtered data
        T_new=data_inf['tiempo'].to_numpy()
        rt_new=data_inf['Rt'].to_numpy()
        rt_new=rt_new-np.mean(rt_new) # Centered covariable
        #rt_new=np.ones(len(T_new))
        rt_and_T=np.column_stack([rt_new,T_new])
        hazard_component_new=np.log(data_inf.hazard.to_numpy())
        # Linear regression
        reg=LinearRegression(fit_intercept=True).fit(rt_and_T,hazard_component_new)
        if intercept:
            parameters[index]=np.array([reg.intercept_,reg.coef_[0],reg.coef_[1]])
        else:
            parameters[index]=reg.coef_
        #reg_intercept=reg.coef_[0]*mean_T_new
        #print(reg.intercept_)
        index+=1
    parameters['prob_a_priori']=np.exp(gm[0]._estimate_log_weights())
    # Dictionary with the whole initial parameter vector
    return parameters
'''
Obtain initial conditions for all the parameters using non parametric estimation of risk with Rt (Rt not centered)
'''
def estimate_initial_parameters_risk_Rt_not_centered(G,data_sample,data_state,intercept=False):
    X=np.column_stack([data_sample['tiempo'].to_numpy(),data_sample['Rt'].to_numpy()])
    # dias=np.asarray(data_state.index)
    gm=GMM(G,X)
    # Clusters
    data_sample['cluster']=gm[0].predict(X)
    data_state['cluster']=gm[0].predict(np.column_stack([data_state.index.to_numpy(),data_state.Rt.to_numpy()]))
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    '''
    Nelson-Aalen estimation
    '''
    '''
    T=data_sample['tiempo'].to_numpy()
    E=np.ones(len(T))
    naf=NelsonAalenFitter()
    naf.fit(T,event_observed=E)
    smoothed_hazard_sample=naf.smoothed_hazard_(3.0)
    smoothed_hazard_sample['timeline']=smoothed_hazard_sample.index
    smoothed_hazard_sample=smoothed_hazard_sample.sort_values('timeline')
    timeline=smoothed_hazard_sample['timeline'].to_numpy()
    hazard=smoothed_hazard_sample['differenced-NA_estimate'].to_numpy()
    hazard_function=inter_soft(timeline,hazard,3)
    '''
    # Data and parameters per cluster
    data_clusters={}
    parameters={}
    index=0
    for i in labels:
        data_clusters[index]=data_sample.loc[data_sample['cluster']==i]
        '''
        Nelson-Aalen estimation
        '''  
        T=data_clusters[index]['tiempo'].to_numpy()
        E=np.ones(len(T))
        naf=NelsonAalenFitter()
        naf.fit(T,event_observed=E)
        smoothed_hazard_sample=naf.smoothed_hazard_(3.0)
        smoothed_hazard_sample['timeline']=smoothed_hazard_sample.index
        smoothed_hazard_sample=smoothed_hazard_sample.sort_values('timeline')
        timeline=smoothed_hazard_sample['timeline'].to_numpy()
        hazard=smoothed_hazard_sample['differenced-NA_estimate'].to_numpy()
        hazard_function=inter_soft(timeline,hazard,3)
        rt=data_clusters[index]['Rt'].to_numpy()
        hazard_component=hazard_function(T)
        # Filter -inf
        dict_inf={'tiempo':T,'hazard':hazard_component,'Rt':rt}
        data_inf=pd.DataFrame(data=dict_inf)
        data_inf=data_inf[data_inf['hazard']>0.001]
        data_inf=data_inf[data_inf['Rt']>0.001]
        # Filtered data
        T_new=data_inf['tiempo'].to_numpy()
        rt_new=data_inf['Rt'].to_numpy()
        rt_new=rt_new#-np.mean(rt_new) # Centered covariable
        #rt_new=np.ones(len(T_new))
        rt_and_T=np.column_stack([rt_new,-T_new])
        hazard_component_new=np.log(data_inf.hazard.to_numpy())
        # Linear regression
        reg=LinearRegression(fit_intercept=True).fit(rt_and_T,hazard_component_new)
        if intercept:
            parameters[index]=np.array([reg.intercept_,reg.coef_[0],reg.coef_[1]])
        else:
            parameters[index]=reg.coef_
        #reg_intercept=reg.coef_[0]*mean_T_new
        #print(reg.intercept_)
        index+=1
    parameters['prob_a_priori']=np.exp(gm[0]._estimate_log_weights())
    # Dictionary with the whole initial parameter vector
    return parameters
'''
Obtain Rt coefficient through linear regression (Cum hazard)
'''
def estimate_parameters_Rt_cum_hazard(G,data_sample,data_state,cs_cum_hazard_par):
    #data_sample=data_sample.sample(frac=0.2)
    X=np.column_stack([data_sample.tiempo.to_numpy(),data_sample['Rt'].to_numpy()])
    # dias=np.asarray(data_state.index)
    gm=GMM(G,X)
    # Clusters
    data_sample['cluster']=gm[0].predict(X)
    data_state['cluster']=gm[0].predict(np.column_stack([data_state.index.to_numpy(),data_state.Rt.to_numpy()]))
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    '''
    Nelson-Aalen estimation
    '''
    '''
    T=data_sample['tiempo'].to_numpy()
    E=np.ones(len(T))
    naf=NelsonAalenFitter()
    naf.fit(T,event_observed=E)
    smoothed_hazard_sample=naf.smoothed_hazard_(3.0)
    smoothed_hazard_sample['timeline']=smoothed_hazard_sample.index
    smoothed_hazard_sample=smoothed_hazard_sample.sort_values('timeline')
    timeline=smoothed_hazard_sample['timeline'].to_numpy()
    hazard=smoothed_hazard_sample['differenced-NA_estimate'].to_numpy()
    hazard_function=inter_soft(timeline,hazard,3)
    '''
    # Data and parameters per cluster
    data_clusters={}
    parameters=np.empty(len(labels))
    #intercept=np.empty(len(labels))
    index=0
    for i in labels:
        data_clusters[index]=data_sample.loc[data_sample['cluster']==i].sample(frac=1)
        '''
        Nelson-Aalen estimation
        '''  
        T=data_clusters[index]['tiempo'].to_numpy()
        E=np.ones(len(T))
        naf=NelsonAalenFitter()
        naf.fit(T,event_observed=E)
        cum_hazard_sample=naf.cumulative_hazard_
        cum_hazard_sample=cum_hazard_sample.reset_index(drop=False)
        cum_hazard_sample=cum_hazard_sample.sort_values('timeline')
        cum_hazard_sample=cum_hazard_sample[cum_hazard_sample['timeline']>0.0]
        timeline=cum_hazard_sample['timeline'].to_numpy()
        cum_hazard=cum_hazard_sample['NA_estimate'].to_numpy()
        cum_hazard_function=inter_soft(timeline,cum_hazard,3)
        rt=data_clusters[index]['Rt'].to_numpy()
        '''
        print('Sample size: ',data_clusters[index].shape[0])
        print('Max-min timeline: ',np.max(timeline),np.min(timeline))
        print('Max-min T: ',np.max(T),np.min(T))
        print('\n')
        '''
        cum_hazard_component=cum_hazard_function(T)
        # Filter -inf
        dict_inf={'tiempo':T,'cum_hazard':cum_hazard_component,'Rt':rt}
        data_inf=pd.DataFrame(data=dict_inf)
        data_inf=data_inf[data_inf['cum_hazard']>0.001]
        data_inf=data_inf[data_inf['Rt']>0.01]
        # Data for regression
        T_new=data_inf['tiempo'].to_numpy()
        rt_new=data_inf['Rt'].to_numpy()-np.mean(data_inf['Rt'].to_numpy())
        X=np.column_stack([rt_new])
        log_cum_hazard_component_non_par=np.log(data_inf.cum_hazard.to_numpy())
        cum_hazard_component_par=cs_cum_hazard_par(T_new)
        log_cum_hazard_component_par=np.log(cum_hazard_component_par)
        Y=log_cum_hazard_component_non_par-log_cum_hazard_component_par
        # Linear regression
        reg=LinearRegression(fit_intercept=False).fit(X,Y)
        parameters[index]=reg.coef_[0]
        #intercept[index]=reg.intercept_
        index+=1
    # Dictionary with the whole initial parameter vector
    return parameters,labels,gm
'''
Obtain Rt coefficient through linear regression (hazard)
'''
def estimate_parameters_Rt_hazard(G,data_sample,data_state,cs_hazard_par):
    X=np.column_stack([data_sample.tiempo.to_numpy(),data_sample['Rt'].to_numpy()])
    gm=GMM(G,X)
    # Clusters
    data_sample['cluster']=gm[0].predict(X)
    data_state['cluster']=gm[0].predict(np.column_stack([data_state.index.to_numpy(),data_state.Rt.to_numpy()]))
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    # Data and parameters per cluster
    data_clusters={}
    parameters=np.empty(len(labels))
    index=0
    for i in labels:
        data_clusters[index]=data_sample.loc[data_sample['cluster']==i].sample(frac=1)
        '''
        Nelson-Aalen estimation
        '''  
        T=data_clusters[index]['tiempo'].to_numpy()
        E=np.ones(len(T))
        naf=NelsonAalenFitter()
        naf.fit(T,event_observed=E)
        hazard_sample=naf.smoothed_hazard_(3.0)
        hazard_sample['timeline']=hazard_sample.index
        hazard_sample=hazard_sample[hazard_sample['timeline']>0.0]
        timeline=hazard_sample['timeline'].to_numpy()
        hazard=hazard_sample['differenced-NA_estimate'].to_numpy()
        hazard_function=inter_soft(timeline,hazard,3)
        rt=data_clusters[index]['Rt'].to_numpy()
        hazard_component=hazard_function(T)

        # Filter -inf
        dict_inf={'tiempo':T,'hazard':hazard_component,'Rt':rt}
        data_inf=pd.DataFrame(data=dict_inf)
        data_inf=data_inf[data_inf['hazard']>0.001]

        # Data for regression
        T_new=data_inf['tiempo'].to_numpy()
        rt_new=data_inf['Rt'].to_numpy()-np.mean(data_inf['Rt'].to_numpy())
        X=np.column_stack([rt_new])
        log_hazard_component_non_par=np.log(data_inf.hazard.to_numpy())
        hazard_component_par=cs_hazard_par(T_new)
        log_hazard_component_par=np.log(hazard_component_par)
        Y=log_hazard_component_non_par-log_hazard_component_par
        # Linear regression
        reg=LinearRegression(fit_intercept=False).fit(X,Y)
        parameters[index]=reg.coef_[0]
        #intercept[index]=reg.intercept_
        index+=1
    # Dictionary with the whole initial parameter vector
    return parameters,labels,gm
'''
Classify by cepas 
'''
# For sample dataframe
def f_sample(row,range_cepas):
    if row['tiempo'] <= range_cepas[0]:
        val = 0
    elif row['tiempo'] > range_cepas[0] and row['tiempo']<=range_cepas[1]:
        val = 1
    else:
        val = 2
    return val
# For state dataframe
def f_state(row,range_cepas):
    if row['tiempo'] <= range_cepas[0]:
        val = 0
    elif row['tiempo'] > range_cepas[0] and row['tiempo']<=range_cepas[1]:
        val = 1
    else:
        val = 2
    return val
'''
Obtain Rt coefficient cepas through linear regression (cum hazard)
'''
def estimate_parameters_Rt_cepas_cum_hazard(data_sample,data_state,range_cepas,cs_cum_hazard_par):
    # Clustering by cepa
    data_sample['cluster']=data_sample.apply(f_sample,range_cepas=range_cepas,axis=1)
    data_state['tiempo']=data_state.index.to_numpy()
    data_state['cluster']=data_state.apply(f_state,range_cepas=range_cepas,axis=1)
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    # Data and parameters per cluster
    data_clusters={}
    parameters=np.empty(len(labels))
    #intercept=np.empty(len(labels))
    index=0
    for i in labels:
        data_clusters[index]=data_sample.loc[data_sample['cluster']==i].sample(frac=1)
        '''
        Nelson-Aalen estimation
        '''  
        T=data_clusters[index]['tiempo'].to_numpy()
        E=np.ones(len(T))
        naf=NelsonAalenFitter()
        naf.fit(T,event_observed=E)
        cum_hazard_sample=naf.cumulative_hazard_
        cum_hazard_sample=cum_hazard_sample.reset_index(drop=False)
        cum_hazard_sample=cum_hazard_sample.sort_values('timeline')
        cum_hazard_sample=cum_hazard_sample[cum_hazard_sample['timeline']>0.0]
        timeline=cum_hazard_sample['timeline'].to_numpy()
        cum_hazard=cum_hazard_sample['NA_estimate'].to_numpy()
        cum_hazard_function=inter_soft(timeline,cum_hazard,3)
        rt=data_clusters[index]['Rt'].to_numpy()
        '''
        print('Sample size: ',data_clusters[index].shape[0])
        print('Max-min timeline: ',np.max(timeline),np.min(timeline))
        print('Max-min T: ',np.max(T),np.min(T))
        print('\n')
        '''
        cum_hazard_component=cum_hazard_function(T)
        # Filter -inf
        dict_inf={'tiempo':T,'cum_hazard':cum_hazard_component,'Rt':rt}
        data_inf=pd.DataFrame(data=dict_inf)
        data_inf=data_inf[data_inf['cum_hazard']>0.001]
        data_inf=data_inf[data_inf['Rt']>0.01]
        # Data for regression
        T_new=data_inf['tiempo'].to_numpy()
        rt_new=data_inf['Rt'].to_numpy()-np.mean(data_inf['Rt'].to_numpy())
        X=np.column_stack([rt_new])
        log_cum_hazard_component_non_par=np.log(data_inf.cum_hazard.to_numpy())
        cum_hazard_component_par=cs_cum_hazard_par(T_new)
        log_cum_hazard_component_par=np.log(cum_hazard_component_par)
        Y=log_cum_hazard_component_non_par-log_cum_hazard_component_par
        # Linear regression
        reg=LinearRegression(fit_intercept=False).fit(X,Y)
        parameters[index]=reg.coef_[0]
        #intercept[index]=reg.intercept_
        index+=1
    # Dictionary with the whole initial parameter vector
    return parameters,labels
'''
Obtain Rt coefficient cepas through linear regression (hazard)
'''
def estimate_parameters_Rt_cepas_hazard(data_sample,data_state,range_cepas,cs_hazard_par):
    # Clustering by cepa
    data_sample['cluster']=data_sample.apply(f_sample,range_cepas=range_cepas,axis=1)
    data_state['tiempo']=data_state.index.to_numpy()
    data_state['cluster']=data_state.apply(f_state,range_cepas=range_cepas,axis=1)
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    # Data and parameters per cluster
    data_clusters={}
    parameters=np.empty(len(labels))
    #intercept=np.empty(len(labels))
    index=0
    for i in labels:
        data_clusters[index]=data_sample.loc[data_sample['cluster']==i].sample(frac=1)
        '''
        Nelson-Aalen estimation
        '''  
        T=data_clusters[index]['tiempo'].to_numpy()
        E=np.ones(len(T))
        naf=NelsonAalenFitter()
        naf.fit(T,event_observed=E)
        hazard_sample=naf.smoothed_hazard_(3.0)
        hazard_sample['timeline']=hazard_sample.index
        hazard_sample=hazard_sample[hazard_sample['timeline']>0.0]
        timeline=hazard_sample['timeline'].to_numpy()
        hazard=hazard_sample['differenced-NA_estimate'].to_numpy()
        hazard_function=inter_soft(timeline,hazard,3)
        rt=data_clusters[index]['Rt'].to_numpy()
        hazard_component=hazard_function(T)

        # Filter -inf
        dict_inf={'tiempo':T,'hazard':hazard_component,'Rt':rt}
        data_inf=pd.DataFrame(data=dict_inf)
        data_inf=data_inf[data_inf['hazard']>0.001]
        
        # Data for regression
        T_new=data_inf['tiempo'].to_numpy()
        rt_new=data_inf['Rt'].to_numpy()-np.mean(data_inf['Rt'].to_numpy())
        X=np.column_stack([rt_new])
        log_hazard_component_non_par=np.log(data_inf.hazard.to_numpy())
        hazard_component_par=cs_hazard_par(T_new)
        log_hazard_component_par=np.log(hazard_component_par)
        Y=log_hazard_component_non_par-log_hazard_component_par
        # Linear regression
        reg=LinearRegression(fit_intercept=False).fit(X,Y)
        parameters[index]=reg.coef_[0]
        #intercept[index]=reg.intercept_
        index+=1
    # Dictionary with the whole initial parameter vector
    return parameters,labels
'''
Obtain initial conditions for all parameters using emperic estimation 
'''
def estimate_initial_parameters_empiric(G,data_state,data_sample):
    X=np.column_stack([data_sample['tiempo'].to_numpy()])
    gm=GMM(G,X)
    # Clusters
    data_sample['cluster']=gm[0].predict(X)
    data_state['cluster']=gm[0].predict(np.column_stack([data_state.index.to_numpy()]))
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    # Separate data in clusters
    collection_cluster={}
    for i in range(len(labels)):
        if i==0:
            collection_cluster[i]=data_state[data_state.cluster==labels[i]]
        else:
            last_day_accumulated=collection_cluster[i-1].iloc[-1]['CasosAcumulados']
            collection_cluster[i]=data_state[data_state.cluster==labels[i]]
            old_accumulated_cases=collection_cluster[i]['CasosAcumulados']
            collection_cluster[i]['CasosAcumulados']=old_accumulated_cases-last_day_accumulated
    parameters={}
    # Calculate parameters per component
    for i in range(len(labels)):
        # Empiric hazard function
        hazard_empiric=collection_cluster[i].CasosDiarios.to_numpy()/(collection_cluster[i].CasosAcumulados.to_numpy())
        hazard_function=inter_soft(collection_cluster[i].index.to_numpy(),hazard_empiric,3)
        # Data Sample of the same cluster
        data_sample_cluster=data_sample[data_sample.cluster==labels[i]]
        time=data_sample_cluster.tiempo.to_numpy()
        dict_inf={'tiempo':time,'hazard':hazard_function(time)}
        data_inf=pd.DataFrame(dict_inf)
        data_inf=data_inf[data_inf.hazard>1e-6]
        data_inf=data_inf[data_inf.tiempo>1]
        # Linear Regression Gompertz parameters
        new_time=data_inf.tiempo.to_numpy()
        mean_new_time=np.mean(new_time)
        log_hazard_component=np.log(data_inf.hazard.to_numpy())
        mat_T=np.column_stack([new_time])
        reg=LinearRegression(fit_intercept=True).fit(mat_T,log_hazard_component)
        #reg_intercept_=reg.coef_[0]*mean_new_time
        parameters[i]=np.array([reg.intercept_,reg.coef_[0]])
    # A priori probabilities 
    parameters['prob_a_priori']=np.exp(gm[0]._estimate_log_weights())
    return parameters
'''
Getting Gompertz Parameters per component
'''
def neg_gomp_IC(G,data_state,data_sample,population):
    #data_sample=data_sample.sample(frac=0.2)
    X=np.column_stack([data_sample['tiempo'].to_numpy()])
    gm=GMM(G,X)
    # Clusters
    data_sample['cluster']=gm[0].predict(X)
    data_state['cluster']=gm[0].predict(np.column_stack([data_state.index.to_numpy()]))
    # Labels that conserve the cluster order
    df_aux=data_state.drop_duplicates(subset='cluster')
    labels=df_aux['cluster'].to_list()
    # Separate data in clusters
    collection_cluster={}
    for i in range(len(labels)):
        if i==0:
            collection_cluster[i]=data_state[data_state.cluster==labels[i]].reset_index(drop=True)
        else:
            last_day_accumulated=collection_cluster[i-1].iloc[-1]['CasosAcumulados']
            collection_cluster[i]=data_state[data_state.cluster==labels[i]].reset_index(drop=True)
            old_accumulated_cases=collection_cluster[i]['CasosAcumulados']
            collection_cluster[i]['CasosAcumulados']=old_accumulated_cases-last_day_accumulated
    # Negative Gompertz parameters per component
    parameters={}
    for i in range(len(labels)):
        t=collection_cluster[i].index.to_numpy()
        num_cum=collection_cluster[i].CasosAcumulados.to_numpy()
        params_gompertz = Parameters()
        # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
        if i>0:
            params_gompertz.add_many(('N_max', num_cum[-1], True, 0, population-collection_cluster[i-1].CasosAcumulados.to_numpy()[-1], None, None),
            ('beta', 10.0, True, 0, None, None, None),
            ('kappa', 0.02, True, 1e-6, 0.9, None, None))
        else:
            params_gompertz.add_many(('N_max', num_cum[-1], True, 0, population, None, None),
            ('beta', 10.0, True, 0, None, None, None),
            ('kappa', 0.02, True, 1e-6, 0.9, None, None))
        minner=Minimizer(residuals_gompertz,params_gompertz,fcn_args=(t,num_cum))
        fit_gompertz=minner.minimize()
        parameters[i]=fit_gompertz.params.valuesdict()
        kappa=parameters[i].pop('kappa')
        beta=parameters[i].pop('beta')
        gamma=np.log(kappa*beta)
        parameters[i]['gamma']=gamma
        parameters[i]['kappa']=kappa
    # A priori probabilities 
    parameters['prob_a_priori']=np.exp(gm[0]._estimate_log_weights())
    return parameters
'''
Getting cum hazard through hazard
'''
def cum_hazard_(days,hazard):
    cs_hazard=CubicSpline(days,hazard)
    cum_hazard=np.zeros(len(days))
    aux=quad(cs_hazard,0.0,days[0])
    cum_hazard[0]=aux[0]
    for j in range(1,len(days)):
        aux=quad(cs_hazard,days[j-1],days[j])
        cum_hazard[j]=cum_hazard[j-1]+aux[0]
    return cum_hazard
'''
NLLS for optimal number of susceptibles
'''
# Residual ejemplo 
def R(z,paramf,P,Y0):
    x,y=paramf.T
    value=(Y0-P/z)*x-y
    return value

# Matriz Jacobiana ejemplo
def J(z,paramf,P):
    x,y=paramf.T
    jac=np.empty((len(x),1))
    jac[:,0]=(P/z**2)*x
    return jac

# Levenberg-Marquart Mínimos Cuadrados No lineales
def levenberg_marquardt_nlls(R,J,z0,N,tol,mu_ref,paramf,P,Y0):
    res=0 # Si se queda en ese valor el algoritmo no converge
    zk,k=z0,0
    Rk_old=R(z0,paramf,P,Y0)
    Jk_old=J(z0,paramf,P)
    fk_old=0.5*Rk_old.T@Rk_old
    A, g = Jk_old.T@Jk_old, Jk_old.T@Rk_old
    mu=np.min([mu_ref,np.max(np.diag(A))])
    while k<N:
        pk=np.linalg.solve(A+mu*np.eye(A.shape[0]),-g)
        if np.linalg.norm(pk)<tol:
            res=1
            break
        '''
        k+1 data
        '''
        k+=1
        zk=zk+pk
        Rk_new=R(zk,paramf,P,Y0)
        fk_new=0.5*Rk_new.T@Rk_new
        '''
        parametro rho
        '''
        denominator=np.squeeze(-0.5*pk.T@g+0.5*mu*pk.T@pk)
        rho=(fk_new-fk_old)/denominator
        if rho<0.25:
            mu=2.0*mu
        elif rho>0.75:
            mu=mu/3.0
        Jk=J(zk,paramf,P)
        A, g = Jk.T@Jk, Jk.T@Rk_new
    
    return {'zk':zk,'fk':fk_new,'k':k,'|pk|':np.linalg.norm(pk),'res':res}

# Prueba del algoritmo de Levenberg-Marquart
def proof_levenberg_marquardt_nlls(R,J,z0,N,tol,mu_ref,paramf,P,Y0,date):
    dic_results=levenberg_marquardt_nlls(R,J,z0,N,tol,mu_ref,paramf,P,Y0)
    if dic_results['res']==0:
        zk=dic_results['zk']
        Rk=R(zk,paramf,P,Y0)
        norm_pk=dic_results['|pk|']
        print('Levenberg-Marquardt algorithm NO CONVERGED')
        print('z0 = ',z0)
        R0=R(z0,paramf,P,Y0)
        f0=0.5*R0.T@R0
        print('f(z0) = ',f0)
        print('zk = ',zk)
        fk=0.5*Rk.T@Rk
        print('f(zk) = ',fk)
        print('|pk| = ',norm_pk)
        print('k = ',dic_results['k'])
    else:
        zk=dic_results['zk']
        Rk=R(zk,paramf,P,Y0)
        norm_pk=dic_results['|pk|']
        print('Levenberg-Marquardt algorithm CONVERGED')
        print('z0 = ',z0)
        R0=R(z0,paramf,P,Y0)
        f0=0.5*R0.T@R0
        print('f(z0) = ',f0)
        print('zk = ',zk)
        fk=0.5*Rk.T@Rk
        print('f(zk) = ',fk)
        print('|pk| = ',norm_pk)
        print('k = ',dic_results['k'])
        '''
        Gráfica del ajuste obtenido
        '''
        # Date format
        date_format = DateFormatter('%Y-%m-%d')
        x,y=paramf.T
        y_model=(Y0-P/zk)*x
        plt.style.use('seaborn')
        plt.plot_date(date,y,'k-',label='observed')
        plt.plot_date(date,y_model,'g-',label='mean')
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.title('Poisson mean for daily infected')
        plt.ylabel('infected')
        plt.xlabel('date')
        plt.legend()
        plt.show()
    return dic_results
