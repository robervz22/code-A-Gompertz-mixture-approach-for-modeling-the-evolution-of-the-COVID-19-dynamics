import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.stats import expon,uniform
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
import random

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
Soft with Maxima
'''
def inter_soft_max(X, Y, step):
    interval = mt.floor(len(Y)/step)
    Y_inter = []
    if len(Y) % step == 0:
        Y_inter = np.empty(interval)
        X_inter = np.empty(interval)
        for i in range(interval):
            index = np.argmax(Y[step*i:step*(i+1)])
            Y_inter[i] = Y[step*i+index]
            X_inter[i] = X[step*i+index]
        if (interval-1)*step < len(Y)-1:
            Y_inter = np.append(Y_inter, Y[-1])
            X_inter = np.append(X_inter, X[-1])
        x = X_inter
        y = Y_inter
        cs_A = CubicSpline(x, y)
    else:
        Y_inter = np.empty(interval+1)
        X_inter = np.empty(interval+1)
        for i in range(interval):
            index = np.argmax(Y[step*i:step*(i+1)])
            Y_inter[i] = Y[step*i+index]
            X_inter[i] = X[step*i+index]
        index = np.argmax(Y[step*interval:])
        Y_inter[interval] = Y[step*interval+index]
        X_inter[interval] = X[step*interval+index]
        if (interval)*step < len(Y)-1:
            Y_inter = np.append(Y_inter, Y[-1])
            X_inter = np.append(X_inter, X[-1])
        x = X_inter
        y = Y_inter
        cs_A = CubicSpline(x, y)
    return cs_A
'''
Mobile Mean
'''
def mobile_mean(X,Y,step):
    tmp=step-1
    X_inter=np.empty(len(X))
    Y_inter=np.empty(len(Y))
    X_inter=X.copy()
    Y_inter[:tmp]=Y[:tmp]
    # First 0,1,step-2 index
    '''
    for i in range(tmp):
        X_inter[i]=X[i]
        Y_inter[i]=Y[i]
    '''
    Y_inter[tmp:]=np.array([np.mean(Y[i-tmp:i+1]) for i in range(tmp,len(Y))])
    '''
    for i in range(tmp,len(Y)):
        X_inter[i]=X[i]
        first_index=i-tmp
        Y_inter[i]=np.mean(Y[first_index:i+1])
    '''
    cs_mobile_mean=CubicSpline(X_inter,Y_inter)
    return cs_mobile_mean
'''
General Sample Class
'''
class g_sample:
    # Initialized
    def __init__(self, data, path_file, size, m_mean):
        self.path_file = path_file
        self.num_dia = data['CasosDiarios'].to_numpy()
        self.num_cum = data['CasosAcumulados'].to_numpy()
        self.dias = data.index.to_numpy()
        self.nrow = len(self.num_dia)

        # Daily infected mobile mean
        cs_num_dia_mobile_mean = mobile_mean(self.dias, self.num_dia, m_mean)
        self.num_dia_mean = cs_num_dia_mobile_mean(self.dias)

        # Cumulative infected mobile mean
        num_cum_mean = np.empty(len(self.num_dia_mean))
        num_cum_mean[0] = self.num_dia_mean[0]
        for i in range(1, self.nrow):
            num_cum_mean[i] = num_cum_mean[i-1]+self.num_dia_mean[i]

        self.num_cum_mean = num_cum_mean

        N0 = mt.floor(self.num_cum_mean[0])  # Número inicial de infectados
        N1 = mt.floor(self.num_cum_mean[-1])  # Número final de infectados
        N = mt.floor((N1-N0)/100.0)
        self.shift = mt.floor((N1-N0)/N)
        self.sample_Select = np.random.uniform(N0, N0+self.shift, size)
        for i in range(N-1):
            sample_SelectAux = np.random.uniform(
                N0+(i+1)*self.shift+1, N0+(i+2)*self.shift, size)
            self.sample_Select = np.concatenate(
                (self.sample_Select, sample_SelectAux))
        self.time = np.empty(len(self.sample_Select))
    # Search the correspond day through binary search
    def searchDay(self):
        day = 0
        for i in range(len(self.sample_Select)):
            individuo = self.sample_Select[i]
            n = len(self.num_cum_mean)-1
            ext_inf = 0
            ext_sup = n
            k = mt.floor((ext_inf+ext_sup)/2)
            while individuo <= self.num_cum_mean[k] or individuo > self.num_cum_mean[k+1]:
                if individuo > self.num_cum_mean[k]:
                    ext_inf = k
                if individuo < self.num_cum_mean[k]:
                    ext_sup = k
                if individuo == self.num_cum_mean[k]:
                    break
                k = mt.floor((ext_inf+ext_sup)/2)
            day = k+1
            if day>=1:
                self.time[i] = np.random.randint(self.dias[day-1],self.dias[day]+1)
            else:
                self.time[i]= self.dias[day]
    # Obtain the sample data file
    def main(self):
        self.searchDay()
        d = {'tiempo': self.time}
        newData = pd.DataFrame(data=d)
        newData.to_csv(self.path_file)
    # Get the sample data frame
    def get_sampling_data(self):
        d = {'tiempo': self.time}
        newData = pd.DataFrame(data=d)
        return newData
'''
General Samples States
'''
class g_samples_states:
    # Initialize
    def __init__(self, path_samples, list_of_states):
        self.path_samples = path_samples
        self.list_of_states=list_of_states
        # Observed data
        self.general_sample_data = [path_samples+'General_Sample_' + state+'.csv' for state in list_of_states]
        self.states_samples_dic = {'g_sample': self.general_sample_data}
        self.states_samples_dic = pd.DataFrame(data=self.states_samples_dic)
        self.states_samples_dic.index = list_of_states
    # Obtain samples
    def generate_samples(self,collection_states,SAMPLES,M_MEAN):
        # Sample and Rt relation
        for state in self.list_of_states:
            try:
                print(f'Process: {state}')
                # Sampling observed data
                df_test=collection_states[state].copy()
                gs=g_sample(df_test,self.states_samples_dic.loc[state]['g_sample'],SAMPLES,M_MEAN)
                gs.main()
                gs_=gs.get_sampling_data()
                # Create .csv file
                gs_.to_csv(self.states_samples_dic.loc[state]['g_sample'])
                print('-----Done-----')
            except Exception as e:
                print(f'Process: {state} Failed')
                print(e)
                pass
    # Visualization
    def sample_visual(self,collection_states,M_MEAN):
        # Sample and Daily Infected per Mexico state
        fig, axs=plt.subplots(8,4,figsize=(15,15))
        index_state=0
        for i in range(8):
            for j in range(4):
                state=self.list_of_states[index_state]
                aux1=collection_states[state].copy()
                cs_aux1=mobile_mean(aux1.index,aux1['CasosDiarios'].to_numpy(),M_MEAN)
                aux2=pd.read_csv(self.states_samples_dic.loc[state]['g_sample'])
                axs[i,j].plot(aux1.index,cs_aux1(aux1.index),color='blue')
                axs[i,j].hist(aux2['tiempo'],orientation='vertical',alpha=0.3,bins=50)
                axs[i,j].set_title(state)
                index_state=index_state+1
        for ax in axs.flat:
            ax.set(xlabel='days', ylabel='infected')
        for ax in axs.flat:
            ax.label_outer()
        # set the spacing between subplots
        plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
'''
Metropolis-Hastings sampling
'''
class MH_sample:
    # Initialized
    def __init__(self, data, path_file, size, m_mean,sd):
        # Charge parameters into class
        self.path_file = path_file
        self.num_dia = data['CasosDiarios'].to_numpy()
        self.num_cum = data['CasosAcumulados'].to_numpy()
        self.dias = np.asarray(data.index,dtype=np.float64)
        self.nrow = len(self.num_dia)
        self.size=size
        self.sd=sd
        # Daily infected mobile mean
        cs_num_dia_mobile_mean = mobile_mean(self.dias, self.num_dia, m_mean)
        self.num_dia_mean = cs_num_dia_mobile_mean(self.dias)

        # Cumulative infected mobile mean
        num_cum_mean = np.empty(len(self.num_dia_mean))
        num_cum_mean[0] = self.num_dia_mean[0]
        for i in range(1, self.nrow):
            num_cum_mean[i] = num_cum_mean[i-1]+self.num_dia_mean[i]

        self.num_cum_mean = num_cum_mean

        # Density estimation
        self.density_emp=num_cum_mean[0]*self.num_dia_mean/(self.num_cum_mean**2)
    # Pi proposal
    def pi_proposal(self,x,y): #pi(x|y)
        # Caso 1
        if y<=self.dias[-1]:
            return np.squeeze(uniform.pdf(x,loc=y,scale=self.sd))
        # Caso 2
        elif y<self.dias[-1] and self.dias[-1]<y+self.sd:
            return np.squeeze(uniform.pdf(x,loc=y,scale=self.dias[-1]-y))
        # Caso 3
        elif self.dias[-1]<y:
            diff=y-self.dias[-1]
            return np.squeeze(uniform.pdf(x,loc=self.dias[-1]-diff,scale=diff))
    # Sampling 
    def sampling(self,x0):
        # Empiric as a function
        cs_density_emp=inter_soft(self.dias,self.density_emp,1)
        sample=[x0]
        index=0
        while index<self.size:
            # Proposal sampling
            # Caso 1
            if sample[-1]<=self.dias[-1]:
                z=np.squeeze(np.random.uniform(sample[-1],sample[-1]+self.sd,1))
            # Caso 2
            elif sample[-1]<self.dias[-1] and self.dias[-1]<sample[-1]+self.sd:
                z=np.squeeze(np.random.uniform(sample[-1],self.dias[-1],1))
            # Caso 3
            elif self.dias[-1]<sample[-1]:
                diff=sample[-1]-self.dias[-1]
                z=np.squeeze(np.random.uniform(self.dias[-1]-diff,self.dias[-1],1))
            # Acceptance or Rejection
            aux=cs_density_emp(z)*self.pi_proposal(sample[-1],z)/(cs_density_emp(sample[-1])*self.pi_proposal(z,sample[-1]))
            prob=np.min(np.array([1.0,aux]))
            if prob>0.5:
                sample.append(z)
                index+=1
            else:
                sample.append(sample[-1])
        return sample
            
'''
Clase muestreo kernel
'''
class kde_sample:
    def __init__(self,data,path_file,size):
        self.path_file=path_file
        self.gtime=data['tiempo'].to_numpy()
        self.size=size

        kde1=sm.nonparametric.KDEUnivariate(self.gtime.astype(np.float))
        self.kde1=kde1.fit()

        kde2=KernelDensity(bandwidth=self.kde1.bw) # Método 2
        self.gtime=self.gtime.reshape((len(self.gtime),1))
        self.kde2=kde2.fit(self.gtime)

        sample=self.kde2.sample(n_samples=self.size,random_state=0)
        self.sample=np.transpose(sample)[0]
    
    def survivor(self):
        self.kde1.fit()
        cs_sur=inter_soft(self.kde1.support,self.kde1.sf,7)
        return cs_sur

    def plot_kde_sample(self):
        plt.plot(self.kde1.support,self.kde1.density,color="k")
        plt.hist(self.sample, bins=50, density=True,label="KDE sample",
           orientation='vertical',alpha=0.3)

    def main(self):
        d={'tiempo':self.sample}
        newData=pd.DataFrame(data=d)
        newData.to_csv(self.path_file)



'''
Pruebas
'''
'''
import os
owd=os.getcwd()
CollectionDir=owd+'/Data/Collection States/'
state_for_proof='NUEVO LEÓN'

data_state=pd.read_csv(CollectionDir+state_for_proof+'.csv')

DataMNGomp=owd+'/Data/MNGomp/'
SIZE=300
SD=2.0
M_MEAN=3

MH_state_for_proof=MH_sample(data_state,DataMNGomp,SIZE,M_MEAN,SD)
#sample_MNGomp=np.asarray(MH_state_for_proof.sampling(20.0))

# Visualization of MH Sampling
plt.plot(MH_state_for_proof.dias[50:],MH_state_for_proof.density_emp[50:],color='r',linestyle='--')
#plt.hist(sample_MNGomp,orientation='vertical',alpha=0.2,bins=50,color='b')
plt.title(f'{state_for_proof}\n Densidad Empírica Gompertz-Negativa')
#plt.legend()
plt.show()
'''
