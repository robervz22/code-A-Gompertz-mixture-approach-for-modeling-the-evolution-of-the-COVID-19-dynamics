# In this file we get the Gompertz parameters throughout EM Algorithm
import numpy as np
import pandas as pd
import math as mt
from scipy.interpolate import CubicSpline
from scipy import optimize
from sklearn.utils import resample
from scipy.integrate import quad

'''
Dates Pandas series to num
'''
def convert_dates_to_days(dates, start_date=None, name='Day'):
    if start_date:
        ts0 = pd.Timestamp(start_date).timestamp()
    else:
        ts0 = 0

    return ((dates.apply(pd.Timestamp.timestamp) - 
            ts0)/(24*3600)).rename(name)
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
Mobile mean
'''
def mobile_mean(X,Y,step):
    tmp=step-1
    X_inter=np.empty(len(X))
    Y_inter=np.empty(len(Y))
    X_inter=X
    Y_inter[:tmp]=Y[:tmp]
    Y_inter[tmp:]=np.array([np.mean(Y[i-tmp:i+1]) for i in range(tmp,len(Y))])
    cs_mobile_mean=CubicSpline(X_inter,Y_inter)
    return cs_mobile_mean

'''
Gompertz mixture class
'''
class mgomp:
    def __init__(self,data,nboot,est_in,pi_in,G):
        # Main variables
        self.time = np.sort(data["tiempo"].to_numpy())
        self.size = len(self.time)
        self.G=G
        # Dirichlet parameters
        self.alpha=10*pi_in 
        self.c=sum(self.alpha)
        # Aux variables
        self.htj = np.empty(self.size)
        self.htau = np.empty((self.G, self.size))
        # Status variables 
        self.iter_max=500
        self.nboot=nboot
        self.ifail=0
        self.inter=0
        MB = 100
        # Initial variables
        self.est_in = est_in
        self.pi_in=pi_in
        # Out variables
        self.estout = np.zeros((MB,self.G, 2)) # Gompertz Parameters
        self.llout = np.empty(MB) # Likelihood
        self.piout=np.zeros((MB,self.G))
        # Iterable variables
        self.est = np.empty((self.G,2))
        self.newest = np.empty((self.G,2))
        # A priori probabilities, A posteriori probabilities and survival function
        self.pi=np.empty(self.G)
        for i in range(self.G):
            self.pi[i] = self.pi_in[i]
        self.tau = np.empty((self.G, self.size))
        self.s = np.empty((self.G, self.size))
        # Component density and Unconditional density
        self.gen_density=np.empty(self.size)
        self.comp_density=np.empty((self.G,self.size))
        # Complete Sample
        self.sampleComplete=np.array([self.time])
        self.sampleComplete=self.sampleComplete.T
    # Get estimators
    def getest(self):
        btime = np.empty(self.size)
        if self.nboot == 1:
            # Without bootstrapping
            self.ifail = 2
            self.mle(self.time, 0)
        else:
            # With Bootstrapping
            for k in range(self.nboot):
                self.ifail = 2
                while self.ifail != 0:
                    self.boot(btime)  # Bootstrapping sample
                    self.mle(btime,k) # Estimation   
    # MLE (EM Algorithm)
    def mle(self, tj, bnum):
        # Status varibles
        self.ifail=0
        self.code = 0
        self.inter = 0
        # Main of the implementation
        done = 0
        for i in range(self.G):
            for l in range(2):
                self.est[i][l] = self.est_in[i][l]
                self.newest[i][l] = self.est[i][l]
        while done != 1 and self.code == 0 and self.inter<self.iter_max:
            # A posteriori probabilities
            self.caltau(tj)
            # Set a priori probabilities
            self.setpi(tj)
            # Gompertz parameters
            self.ifail = 3  # IMPORTANTE
            self.callhy(tj)
            if self.ifail != 0:
                continue
            # Stopping condition
            ddiff = 0.0
            for i in range(self.G):
                for l in range(2):
                    if(self.newest[i][l] != 0):
                        ddiff = ddiff+abs((self.est[i][l]-self.newest[i][l])/self.newest[i][l])
                    self.est[i][l] = self.newest[i][l]
            if ddiff < 0.001:
                done = 1
            self.inter = self.inter+1
        if self.code != 0:
            return
        self.llout[bnum] = self.findll(tj)
        for i in range(self.G):
            for l in range(2):
                self.estout[bnum][i][l] = self.est[i][l]
            self.piout[bnum]=self.pi
        return
    # Survival function for each component
    def calcs(self,tj):
        #t1, t2 = 0.0, 0.0
        udrfl = -75.0
        for i in range(self.G):
            gamma= self.est[i][0]
            kappa= self.est[i][1]
            tmp=-kappa*tj
            aux=(np.exp(gamma+tmp)-np.exp(gamma))/kappa
            self.s[i]=np.array([np.exp(a) if a>udrfl else 0.0 for a in aux])
    # Density
    def caldensity(self,tj):
        self.calcs(tj)
        # Density per component
        for i in range(self.G):
            gamma=self.est[i][0]
            kappa=self.est[i][1]
            tmp=-kappa*tj
            hazard=np.exp(gamma+tmp)
            self.comp_density[i]=hazard*self.s[i]
        pi_row=self.pi.reshape(1,-1)
        self.gen_density=np.squeeze(pi_row@self.comp_density)
    # Posterior probabilities
    def caltau(self,tj):
        self.caldensity(tj)
        for i in range(self.G):
            self.tau[i]=self.pi[i]*self.comp_density[i]/self.gen_density      
    # Set a priori probabilities (EM estimation)
    def setpi(self,tj):
        self.caltau(tj)
        for i in range(self.G):
            t1=sum(self.tau[i,:])
            t2=self.size
            self.pi[i]=t1/t2   
    # BOOTSTRAPPING
    def boot(self,bftime):
        newSample=resample(self.sampleComplete)
        bftime=newSample[:,0]
    # Non linear equation system
    def evalfn(self,x,group):
        tj=self.htj
        tau=self.htau[group]
        fvec = np.zeros(2)
        gamma = x[0]
        kappa = x[1]
        tmp=-kappa*tj
        cval=np.exp(gamma+tmp)
        hval=cval-np.exp(gamma)
        tot1=np.sum(tau*(1.0+hval/kappa))
        aux=tj*(1.0+cval/kappa)
        tot2=np.sum(tau*(aux+hval/kappa**2))
        fvec[0] = tot1
        fvec[1] = tot2
        return fvec
    # Call HYBRD1 (Quasi-Newton Method) 
    def callhy(self,tj):
        # Copies for variables (due to Bootstrapping)
        self.htj=tj
        self.htau=self.tau
        x0 = np.empty(2)
        self.code = 0
        # Defines 2xG equation system
        for group in range(self.G):
            if self.code != 0:
                continue
            gamma = self.est[group][0]
            kappa = self.est[group][1]
            ifault = 0
            x0[0] = gamma
            x0[1] = kappa
            xtol = 1e-6
            try:
                sol=optimize.root(self.evalfn,x0,args=(group),tol=xtol,method='hybr',jac=False)
                ifault=sol.status
            except Exception as e:
                ifault=0
                print(e)
                pass
            # ifault=1 there is not error, change self.code=1 for ifault=0
            if ifault != 1:
                self.code = 1
                if ifault != 1:
                    self.code = ifault
                    self.ifail = 3
                    continue
            gamma=sol.x[0]
            kappa=sol.x[1]
            self.newest[group][0] = gamma
            self.newest[group][1] = kappa
        if self.code == 0:
            self.ifail = 0       
    # Likelihood estimation
    def findll(self, tj):
        self.caldensity(tj)
        ll=np.sum(np.log(self.gen_density))
        return ll
    # Show results in bash
    def output(self):
         # Auxiliars for Gompertz's components
        sum_ = np.zeros((self.G,2))
        bar = np.empty((self.G,2))
        se = np.empty((self.G,2))
        sumsq = np.zeros((self.G,2))
        #Auxiliars for a priori probabilities
        sum_pi=np.zeros(self.G)
        bar_pi=np.empty(self.G)
        se_pi=np.empty(self.G)
        sumsq_pi=np.empty(self.G)
        print("Initial parameters: ")
        print(self.est_in)
        if self.ifail != 0:
            print("The program finished with code error = %d" % (self.ifail))
            if self.ifail == 3:
                print("Linear Equation Error")
                print(self.code)
        else:
            print("\nResults summarize")
            for k in range(self.nboot):
                # Printing without bootstrapping
                if self.nboot == 1:
                    print("Bootstrap= %d, Iter= %d" %
                        (k, self.inter))
                    print('\n')
                    for i in range(self.G):
                        print("Component %d: gamma= %.4f, kappa= %.4f" %
                            (i, self.estout[k][i][0], self.estout[k][i][1]))
                    print("\n")
                    for i in range(self.G):
                        print("Mixture proportion %d: %.4f" %(i,self.piout[k][i]))
                    continue
                # Processing with bootstrapping  <- I am here
                for i in range(self.G):
                    for l in range(2):
                        sum_[i][l] = sum_[i][l]+self.estout[k][i][l]
                        sumsq[i][l] = sumsq[i][l]+self.estout[k][i][l]*self.estout[k][i][l]
                    sum_pi[i]=sum_pi[i]+self.piout[k][i]
                    sumsq_pi[i]=sumsq_pi[i]+self.piout[k][i]*self.piout[k][i]
            if self.nboot == 1:
                return
            for i in range(self.G):
                for l in range(2):
                    bar[i][l] = sum_[i][l]/float(self.nboot)
                    se[i][l] = np.sqrt(sumsq[i][l]/float(self.nboot)-bar[i][l]*bar[i][l])
                bar_pi[i]=sum_pi[i]/float(self.nboot)
                se_pi[i]=np.sqrt(sumsq_pi[i]/float(self.nboot)-bar_pi[i]*bar_pi[i])
            print("\nGompertz component estimators:")
            for i in range(self.G):
                for l in range(2):
                    print("Mean (s.e) of estimator (%d,%d): %.4f (%.4f)" %
                        (i,l, bar[i][l], se[i][l]*float(self.nboot)/float(self.nboot-1)))
            print("\nMixture proportion")
            for i in range(self.G):
                print("Mean (s.e) of mixture proportion component %d: %.4f (%.4f)"%(i,bar_pi[i],se_pi[i]*float(self.nboot)/float(self.nboot-1)))
    # Survival function given time
    def sf(self,t):
        # Survival for each component
        s_comp=np.empty((self.G,len(t)))
        for i in range(self.G):
            if self.nboot==1:
                gamma=self.est[i][0]
                kappa=self.est[i][1]
            else:
                gamma=self.bar[i][0]
                kappa=self.bar[i][1]
            tmp=np.exp(-kappa*t)
            aux=(np.exp(gamma)*(tmp-1.0))/kappa
            s_comp[i]=np.exp(aux)
        # General survival
        s_general=np.empty(len(t))
        pi_row=self.pi.reshape(1,-1)
        s_general=np.squeeze(pi_row@s_comp)
        return s_general,s_comp
    # Growth function basemodel
    def hazard(self,t):
        sf=self.sf(t)
        sf_general=sf[0]
        sf_comp=sf[1]
        self.weights=np.empty((self.G,len(t)))
        for i in range(self.G):
            self.weights[i]=(self.pi[i]*sf_comp[i])/(sf_general+0.001)
        # Hazard per components
        hazard_comp=np.zeros((self.G,len(t)))
        for i in range(self.G):
            if self.nboot==1:
                gamma=self.est[i][0]
                kappa=self.est[i][1]
            else:
                gamma=self.bar[i][0]
                kappa=self.bar[i][1]
            hazard_comp[i]=np.exp(gamma-kappa*t)
        # Hazard mixture
        hazard_general=np.empty(len(t))
        hazard_general=np.sum(self.weights*hazard_comp,axis=0)
        cs_hazard_general=CubicSpline(t,hazard_general)
        # Hazard
        cum_hazard=np.zeros(len(t))
        aux=quad(cs_hazard_general,0.0,t[0])
        cum_hazard[0]=aux[0]
        for j in range(1,len(t)):
            aux=quad(cs_hazard_general,t[j-1],t[j])
            cum_hazard[j]=cum_hazard[j-1]+aux[0]
        return cum_hazard, hazard_general
    # Growth function accelerated model
    def hazard_acc(self,t,rt,beta):
        sf=self.sf(t)
        sf_general=sf[0]
        sf_comp=sf[1]
        self.weights=np.empty((self.G,len(t)))
        for i in range(self.G):
            self.weights[i]=(self.pi[i]*sf_comp[i])/(sf_general+0.001)
        # Hazard per components
        hazard_comp=np.zeros((self.G,len(t)))
        for i in range(self.G):
            if self.nboot==1:
                gamma=self.est[i][0]
                kappa=self.est[i][1]
            else:
                gamma=self.bar[i][0]
                kappa=self.bar[i][1]
            hazard_comp[i]=np.exp(gamma+beta*rt-kappa*t)
        # Hazard mixture
        hazard_general=np.empty(len(t))
        hazard_general=np.sum(self.weights*hazard_comp,axis=0)
        cs_hazard_general=CubicSpline(t,hazard_general)
        # Hazard
        cum_hazard=np.zeros(len(t))
        aux=quad(cs_hazard_general,0.0,t[0])
        cum_hazard[0]=aux[0]
        for j in range(1,len(t)):
            aux=quad(cs_hazard_general,t[j-1],t[j])
            cum_hazard[j]=cum_hazard[j-1]+aux[0]
        return cum_hazard, hazard_general
    # Density given time
    def density(self,t):
        sf=self.sf(t)
        s_comp=sf[1]
        density_comp=np.empty((self.G,len(t)))
        # Density per component
        for i in range(self.G):
            gamma=self.est[i][0]
            kappa=self.est[i][1]
            hazard=np.exp(gamma-kappa*t)
            density_comp[i]=hazard*s_comp[i]
        # General density
        density_gen=np.empty(len(t))
        pi_row=self.pi.reshape(1,-1)
        density_gen=np.squeeze(pi_row@density_comp)
        return density_gen,density_comp
    # A posteriori probabilities given time
    def tau_est(self,t):
        p_gomp=self.density(t)
        density_gen=p_gomp[0]
        density_comp=p_gomp[1]
        tau=np.empty((self.G,len(t)))
        for i in range(self.G):
            tau[i]=(self.pi[i]*density_comp[i])/density_gen
        return tau          
    # Gompertz Clustering 
    def predict(self,t):
        tau=self.tau_est(t)
        predict=[np.argmax(tau[:,j])  for j in range(len(t))]
        return predict





