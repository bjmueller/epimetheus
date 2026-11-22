import numpy as np
from tqdm import tqdm
import copy
import os
import sys

import epimetheus as model


def run_CgR (l2pop=None, rtest=None, Iav_ini = 1e-2, frac_ac1 = None, frac_ac2 = None , win = 40, seed = 11082026):

    rng = np.random.default_rng(seed=seed)
    
    series_CgR = []
    risk_ini = 0.019
    
    for gamma_c in [0.9,1.0,1.1,1.15]:

        exc_contacts_target = 12
        
        res = model.model(l2pop = l2pop, degree_dist = 'uniform', exc_contacts = exc_contacts_target, \
                      alpha_pareto = 2.1, f_cluster = 0.0, c_alpha = 0.5, \
                      gamma_c = gamma_c, gamma_r = 2/9*gamma_c, homogeneous=False, \
                      sus_type = 'uniform', p_sus = (0.4,2.5), p_scal = 0.0, \
                      dur_type = 'uniform', p_dur = 0.2, imm_type = 'uniform', p_imm = (2.5,4), rng = rng)
        
        print ("Iterating to tune average degree.")
        exc_contacts_actual = res.avg_degree
        for iter in range(3):
            if np.abs(exc_contacts_actual-exc_contacts_target)<0.5:
                break
            exc_contacts_new = (exc_contacts_target/exc_contacts_actual) * res.exc_contacts
            res = model.model(l2pop = l2pop, degree_dist = 'uniform', exc_contacts = exc_contacts_new, \
                      alpha_pareto = 2.1, f_cluster = 0.0, c_alpha = 0.5, \
                      gamma_c = gamma_c, gamma_r = 2/9*gamma_c, homogeneous=False, \
                      sus_type = 'uniform', p_sus = (0.4,2.5), p_scal = 0.0, \
                      dur_type = 'uniform', p_dur = 0.2, imm_type = 'uniform', p_imm = (2.5,4))
            exc_contacts_actual = res.avg_degree
            
        mult = 1.0 # Multiplier is unity when we tune the model
        res.tune_rt (r0c = 1.4, tol=0.1, risk_multiplier=risk_ini, waning = (3650,5), \
                 a_seasonality=0.0, nsteps=80, time_dependent_multiplier=mult)    
        
        risk_ini = res.risk_multiplier
        mult = np.ones(280)
        mult[:]=rtest[183:283+180]/1.4
        if gamma_c==0.9:
            n_seed = np.int64(1.5*Iav_ini*2**l2pop)
        elif gamma_c==1.0:
            n_seed = np.int64(0.5*Iav_ini*2**l2pop)
        elif gamma_c==1.1:
            n_seed = np.int64(0.125*Iav_ini*2**l2pop)
        elif gamma_c==1.15:
            n_seed = np.int64(0.1*Iav_ini*2**l2pop)

        for ij in range(3):
            res.simulate (waning = (3650,5), nsteps=280, time_dependent_multiplier=mult, \
                         n_seed = n_seed)
            if np.abs(res.C[-1]/0.089-1) < 0.1:
                break
            n_seed = np.int64 (0.089/res.C[-1] * n_seed)
        
        if frac_ac1 is not None:
            ii = np.where(res.C>frac_ac1*res.C[-1])[0][0]
            res.ac1 = model.autocorrelation(res,ii,window=win)

        if frac_ac2 is not None:
            ii = np.where(res.C>frac_ac2*res.C[-1])[0][0]
            res.ac2 = model.autocorrelation(res,ii,window=win)
              
        series_CgR.append (copy.deepcopy(res))
        series_CgR[-1].clear()

    return series_CgR



def run_series (l2pop = 18, r0 = 1.4, seed = 11082026, risk_ini = 0.019, \
                 frac_ac1 = None, frac_ac2 = None, win =40, **kwargs):

    series_variable  = None
    series_variable2 = None

    rng = np.random.default_rng(seed=seed)
    
    for ii in kwargs.keys():
        if isinstance (kwargs[ii], list):
            if series_variable == None:
                print ("Running series in %s:" % ii)
                print (kwargs[ii])
                series_variable = ii
                series_length = len (kwargs[ii])
            elif series_variable2 == None:
                print ("Second series parameter %s:" % ii)
                print (kwargs[ii])
                series_variable2 = ii
                if len (kwargs[ii]) != series_length:
                    raise Exception ("Parameter lists need to be of identical length.")
            else:
                raise Exception ("A variable parameter for this series is already defined.")
        elif isinstance (kwargs[ii], (int,float,tuple,str)):
            ...
        else:
            raise Exception ("Wrong data type in **kwargs.")

    series = []
    for ii in range(series_length):
        print ("\n\n\n***********************************************\n")
        print ("EPIMETHEUS code running new model in series.")
        
        param = kwargs[series_variable][ii]
        print("Running model with %s = " % series_variable, param)
        kwargs_model = copy.deepcopy(kwargs)
        kwargs_model[series_variable] = param
        if series_variable2 is not None:
            param2 = kwargs[series_variable2][ii]
            kwargs_model[series_variable2] = param2
            print ("Second series parameter %s =" % series_variable2, param2)

        res = model.model(l2pop = l2pop, rng = rng, **kwargs_model)
        try:
            exc_contacts_actual = res.avg_degree
            exc_contacts_target = kwargs_model['exc_contacts']
            iterate_degree_distribution = True
        except:
            iterate_degree_distribution = False
        if iterate_degree_distribution:
            print ("Iterating to tune average degree.")
            for iter in range(3):
                if np.abs(exc_contacts_actual-exc_contacts_target)<0.5:
                    break 
                exc_contacts_new = (exc_contacts_target/exc_contacts_actual) * kwargs_model['exc_contacts']
                kwargs_model['exc_contacts'] = exc_contacts_new
                res = model.model(l2pop = l2pop, rng = rng, **kwargs_model)
                exc_contacts_actual = res.avg_degree

        res.tune_rt (r0c = r0, tol = 0.1, risk_multiplier = risk_ini, waning = (3650,5), \
                     a_seasonality = 0.0, nsteps = 80, **kwargs_model)
        risk_ini = res.risk_multiplier
        res.simulate (waning = (3650,5), nsteps=180)

        if frac_ac1 is not None:
            ii = np.where(res.C>frac_ac1*res.C[-1])[0][0]
            res.ac1 = model.autocorrelation(res,ii,window=win)

        if frac_ac2 is not None:
            ii = np.where(res.C>frac_ac2*res.C[-1])[0][0]
            res.ac2 = model.autocorrelation(res,ii,window=win)
        
        series.append (copy.deepcopy(res))
        series[-1].clear()

    return series
