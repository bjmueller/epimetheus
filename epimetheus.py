import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx
from scipy.stats import logistic
import scipy.sparse as sparse
from scipy.interpolate import Akima1DInterpolator
from scipy.special import lambertw
from tqdm import tqdm
import copy


class model:

    ########################################################################################
    
    def __init__ (self, l2pop = 20, \
                  degree_dist = 'exponential', exc_contacts = 19, \
                  alpha_pareto=2.6, scale_pareto = None, cutoff_pareto = 500, \
                  f_cluster  = 0.0, c_alpha = 1.0, \
                  gamma_c = 0.0, gamma_r = 0, homogeneous = False, \
                  k_tra = None, infectivity = None, dur_type = 'uniform', p_dur = (5,0.2),  \
                  sus_type = 'gamma', p_sus = (5,0.2), p_scal = 0.0, \
                  imm_type = 'uniform', p_imm = 1., cov_imm_sus = 0.0, rng = None, **kwargs):

    # --------------------------------------------------------------------------------------
    # Initialization function for the structure of the transmission network and the
    # distribution of key disease parameters.
    #
    # l2pop: log_2 of the total population npop
    #
    # Network parameters:
    #   degree_dist:  Specifies the distribution of degrees (directed contacts per individual):
    #                 'pareto': (floored) Pareto Type II distribution
    #                 'exponential': exponential distribution
    #                 'poisson': Poisson distribution
    #                 'constant' or unspecified: constant number of directed contacts per individual
    #   alpha_pareto: Tail index for Pareto distributed degrees (must be positive)
    #   exc_contcts:  Expected number of total contacts per individual for expoential
    #   f_cluster:    If > 0, concentrate a fraction f_cluster in the immediate neighborhood
    #   c_alpha:      Sharpness parameter for the clusters (0: maximum concentration)
    #   gamma_c:      Power-law index for the contact probability as a function of distance to
    #                 the lowest common ancestor in the binary tree for the remaining contacts,
    #                 with gamma_c=0 indicating uniform mixing.
    #                 Do not choose extreme values gamma_c>>1 as this also introduces strong
    #                 additional clustering and will result in too few contacts overall.
    #
    #   gamma_r:      Power-law exponent for the spreading kernel as a function of
    #                 scale (distance) in the network
    #
    # Host parameters:
    #   sus_type:     'gamma' for gamma-distributed variations of susceptibility on the
    #                 individual level, else a uniform distribution is used.
    #   p_sus:        Shape and scale for the gamma distribution of susceptibility (tuple).
    #   p_scal:       Scale parameter for lognormal distribution of susceptibility at every
    #                 tree level. Added *independently* of the choice for sus_type.
    #   dur_type:     Distribution of scale factor for duration of immunity. 'gamma' for
    #                 gamma distribution, otherwise the scale factor is set to one.
    #   p_dur:        Parameters for the gamma distribution if dur_type is 'gamma'.
    #
    # --------------------------------------------------------------------------------------

        print ("\n")
        print ("╔═════════════════════════════╗")
        print ("║                             ║")
        print ("║       EPIMETHEUS CODE       ║")
        print ("║                             ║")
        print ("╚═════════════════════════════╝")
        print ("\n")
        
        
        print ("=========================================================================")
        print ("Initialising new model")
        print ("=========================================================================")
        print ("\n")

        if rng is None:
            self.rng = np.random.default_rng(seed=299792458)
        else:
            self.rng = rng
        
        npop = 2**l2pop

        # ----------------------------------------------------------------------------------
        # Set up the transmission network

        use_networkx = False
        
        if degree_dist == 'pareto':
            print ("Using PARETO-like distribution for connectivity.")
            tmp = self.rng.pareto (alpha_pareto, size=npop)
            if scale_pareto is None:
                # For the Pareto distribution, it is more robust to tune the scale
                # factor iteratively instead of using analytic relations:
                ytest = 0.5*exc_contacts-1
                scr0 = ytest / np.average(tmp)
                y0 = np.average (np.minimum(scr0*tmp,cutoff_pareto-1))
                scr1 = scr0 * ytest / y0
                for iter in range(20):
                    y1 = np.average (np.minimum(scr1*tmp,cutoff_pareto-1))
                    slope = np.log(y0/y1)/np.log(scr0/scr1)
                    scr0 = scr1
                    scr1 = scr1 * np.exp(np.log(ytest/y1)/slope)
                    y0 = y1
                    print ("Iterating, (ytest,y1) = (%f5.5,%f5.5)" % (ytest, y1))
                    if np.abs(np.log(ytest/y1))<1e-5:
                        scale_factor = scr1
                        break
                    if iter==19:
                        raise Exception ("Convergence error!")
            else:
                print ("Predefined scale factor for Pareto distribuion, do no tune to exc_caontacts.")
            n_contacts = 1 + np.minimum(scale_factor*tmp,cutoff_pareto-1)
        elif degree_dist == 'exponential':
            print ("Using EXPONENTIAL distribution for connectivity.")
            n_contacts = 1 + self.rng.exponential(exc_contacts/2-1, size=npop)
        elif degree_dist == 'poisson':
            print ("Using POISSON distribution for connectivity.")
            n_contacts = 1+self.rng.poisson(exc_contacts/2-1, size=npop)
        elif degree_dist == 'barabasi-albert':
            print ("Using scale-free Barabasi-Albert network. Overriding settings for gamma_c.")
            use_networkx =True
        else:
            print ("Using CONSTANT number of directed contacts per person.")
            n_contacts = np.int64(exc_contacts/2) * np.ones(npop, dtype = np.int64)

        if not use_networkx:
            k = np.arange(l2pop)
            pop_k = 2 ** k
            K_ij = 1 / pop_k ** gamma_c
            K_ij = K_ij / np.sum (K_ij * pop_k)

            n_rem = f_cluster * n_contacts
            n_contacts_l = np.zeros((npop,l2pop))
            
            if f_cluster>0.0:
                for l in range (l2pop):
                    prob = (n_rem / (f_cluster*n_contacts)) ** c_alpha
                    n_contacts_l[:,l] = np.minimum(n_rem, pop_k[l]*prob)
                    n_rem = n_rem - n_contacts_l[:,l]

            # Distribute remaining contacts according to power law
            pop_k = np.ones(npop)[:,np.newaxis] * pop_k[np.newaxis,:]
            n_non_clustered = pop_k-n_contacts_l
            K_ij_wt = np.ones(npop)[:,np.newaxis] * K_ij
            for i in range(5): # Normalize K_ij, need to iterate if we have gamma_c>1
                K_ij_wt = K_ij_wt * ((1-f_cluster) * n_contacts / \
                                     np.sum (np.minimum(K_ij_wt,1.0) *n_non_clustered, axis=1))[:,np.newaxis]
            for l in range (l2pop):
                prob = np.minimum (K_ij_wt [:,l], 1.0)
                n_contacts_l[:,l] += n_non_clustered[:,l] * prob

            # Rescale n_contacts to n_contacts* such that we get the desired expected number
            # of contacts per level after n_contacts* random draws *with* replacement
            #
            n_contacts_l0 = np.int64 (np.log (np.maximum (1.0 - n_contacts_l/pop_k, 1e-4)) /\
                np.log (np.maximum (1.0 - 1.0/pop_k, 1e-4)))
            n_contacts_l1 = n_contacts_l0 + 1
            y0 = pop_k * (1-(1-1/pop_k)**n_contacts_l0)
            y1 = pop_k * (1-(1-1/pop_k)**n_contacts_l1)
            jj = np.where(self.rng.random (size= (npop, l2pop))>(y1-n_contacts_l)/(y1-y0))
            n_contacts_l [jj] = n_contacts_l1 [jj]
            n_contacts_l = np.int64(n_contacts_l)

            self.n_contacts_l = n_contacts_l
            self.n_contacts_l_exc = np.average(pop_k*(1.0-(1-1.0/pop_k)**n_contacts_l),axis=0)
            n_contacts = np.sum(n_contacts_l,axis=1)
            
            # Now construct the contact matrix, weighted with distance-dependent spreading kernel
            coo_i = []
            coo_j = []
            val   = []

            inx=np.arange(npop, dtype=np.int64)
            for i in tqdm(range(npop)):
                parent_case = i
                dist = np.repeat(k, n_contacts_l[i,:])
                #---
                # may include duplicates
                xx = np.bitwise_xor(parent_case - np.mod(parent_case, 2**dist), 2**dist) + \
                    np.int64(self.rng.uniform(0,2**dist,size=n_contacts[i]))
                xx,jj = np.unique(xx,return_index=True)
                xx = xx.tolist()
                n_contacts_new=len(xx)
                coo_i += xx
                coo_j += (i+np.zeros(n_contacts_new)).tolist()
                val += (1.0/2**(dist[jj]*gamma_r)).tolist()
             
            m_ij = sparse.coo_matrix((val,(coo_i,coo_j)),shape=(npop,npop))
            self.degrees = np.array(np.sum(m_ij.sign(),axis=1))[:,0]
            print("Average number of directed contacts: %f (target %f)" % (np.average(self.degrees), exc_contacts/2))

            m_ij = m_ij.maximum(m_ij.transpose())
            m_ij = m_ij.tocsr()
            
            inx = np.arange(npop, dtype = np.int64)
            self.neighbors = sparse.coo_matrix((inx**0,(inx.tolist(),np.bitwise_xor(inx,1).tolist())),shape=(npop,npop))

        else: # use Networkx package - only Barabasi-Albert network currently implemented
            if degree_dist == 'barabasi-albert':
                g = networkx.barabasi_albert_graph (npop, 1)
                m_ij = networkx.adjacency_matrix (g)
            else:
                raise Exception("degree_dist: unsupported case.")
            coo_i = []
            coo_j = []
            val   = []
            # for each node, pick one contact as neighbor
            for i in tqdm(range(npop)):
                candidates = m_ij.indices [m_ij.indptr[i]:m_ij.indptr[i+1]]
                coo_i += [i]
                coo_j += [self.rng.choice (candidates)]
                val += [1.0]
            self.neighbors = sparse.coo_matrix((val,(coo_i,coo_j)),shape=(npop,npop))
            
        self.degrees = np.array(np.sum(m_ij.sign(),axis=1))[:,0]
        self.avg_degree = np.average(self.degrees)
        print("Average degree: %f" % self.avg_degree)
        
        # ----------------------------------------------------------------------------------
        # Distribution of biological parameters at the individual level

        # Transmissibility multiplier (currently set to 1)
        if k_tra is not None:
            print ("GAMMA distribution for infectivity.")
            tra = self.rng.gamma(k_tra,1/k_tra,size=npop)
        else:
            print ("UNIFORM infectivity.")
            tra = np.ones(npop)

        
        # Susceptibility
        if sus_type == 'gamma':
            print ("GAMMA distribution for susceptibility.")
            sus = self.rng.gamma(p_sus[0],p_sus[1],size=npop)
        else:
            print ("UNIFORM susceptibility.")
            sus = np.ones(npop) # susceptibility

        if p_scal>0.0:
            # Add scale-dependent fluctuations in susceptibility (on top of individual distribution)
            print ("Adding SCALE-DEPENDENT fluctuations to susceptibility.")
            for p in range(l2pop):
                # Note: Some fluctuations on the individual scale added if loop stars from 0
                sus *= np.exp (np.repeat (self.rng.normal(scale=p_scal, size = 2**p), 2**(l2pop-p)))

        # baseline infectivity as a function of age of infection
        if infectivity is None:
            print ("Using preset infectivity function.")
            self.infectivity = np.array([0.0,0.5,0.9,0.9,0.85,0.8,0.7,0.6,0.45,0.15,0.05,0.02,0.0])
        elif infectivity == '4d':
            print ("Constant infectivity for four days.")
            # Non-zero on four days after infection (during ~1 serial interval for COVID-19)
            self.infectivity = np.array([0.,1.,1.,1.,1.,0.])
        else:
            print ("Using customised infectivity function.")
            self.infectivity = np.array(infectivity)
        if self.infectivity[0] != 0.0:
            raise Exception ("Infectivity must be zero on day zero.")
        if self.infectivity[-1] != 0.0:
            raise Exception ("Please include infectivity function to the point when it drops to zero.")
        self.tau = len(self.infectivity)

        self.infectivity_function = Akima1DInterpolator (np.arange(self.tau), self.infectivity)
        if (self.infectivity_function(0) != 0.0) or (self.infectivity_function(self.tau-1) != 0.0):
            raise Exception ("Something wrong with infectivity function.")
                 
        # Scale factor for duration of infectiousness
        if dur_type == 'lognormal':
            print ("LOGNORMAL distribution for infectious period.")
            self.dur = self.rng.lognormal(mean=-p_dur**2/2, sigma=p_dur, size=npop)
        elif dur_type == 'gamma':
            print ("GAMMA distribution for infectious period.")
            self.dur = self.rng.gamma(p_dur[0],p_dur[1],size=npop)
        elif dur_type == 'inf':
            print ("Infinite infectious period for fraction %f of population." % p_dur)
            self.dur = np.ones(npop) + (self.rng.uniform(size=npop)<p_dur) * 20
        else:
            print ("UNIFORM duration of infectious period.")
            self.dur = np.ones(npop)
            dur_type = 'uniform'

        # Average infectivity as a function of time  (*not* weighted by risk of infection and connectivity of primary host)
        if dur_type == 'uniform':
            self.infectivity_avg = np.copy(self.infectivity)
        else:
            self.infectivity_avg = []
            print(self.tau * np.int64(np.ceil(np.max(self.dur))),np.max(self.dur))
            #for time in tqdm (range (self.tau * np.min(100,np.int64(np.ceil(np.max(self.dur)))))):
            for time in tqdm (range (self.tau * np.int64(np.ceil(np.max(self.dur))))):
                age_inf = time
                inf_current_wt = self.infectivity_function (np.int64(np.minimum(age_inf*self.dur,self.tau-1)))
                self.infectivity_avg += [np.average(inf_current_wt)]
            self.infectivity_avg = np.array(self.infectivity_avg)
            
        # Scale factor for duration of immunity
        if imm_type == 'lognormal':
            print ("LOGNORMAL distribution for speed of waning.")
            self.imm = self.rng.lognormal(mean=-p_imm**2/2, sigma=p_dur, size=npop)
        elif imm_type == 'gamma':
            print ("GAMMA distribution for speed of waning.")
            self.imm = self.rng.gamma(p_imm[0],p_imm[1],size=npop)
        else:
            print ("UNIFORM speed of waning.")
            self.imm = np.ones(npop)
            
        # Optional: Let immunity of susceptible individuals wane more quickly
        if cov_imm_sus>0:
            print ("Adding covariance of waning speed with susceptibility.")
            self.imm *= self.sus ** cov_imm_sus
            self.cov_imm_sus = cov_imm_sus

        
        self.l2pop = l2pop
        self.npop = npop

        self.exc_contacts = exc_contacts
        self.degree_dist = degree_dist
        self.alpha_pareto = alpha_pareto
        self.f_cluster = f_cluster
        self.c_alpha = c_alpha
        self.gamma_c = gamma_c 
        self.gamma_r = gamma_r
        self.k_tra = k_tra
        self.homogeneous = homogeneous
        self.sus_type = sus_type
        self.p_sus = p_sus
        self.p_scal = p_scal
        self.dur_type = dur_type
        self.p_dur = p_dur
        self.imm_type = imm_type
        self.p_imm = p_imm

        self.m_ij = m_ij

        self.tra = tra
        self.sus = sus

        print ("Model setup complete.")
        print ("=========================================================================")
        print ("\n")

        
    def simulate(self, nsteps = 250, risk_multiplier = None, n_seed = 20, degree_weighting = True, \
                 intervention = False, npi_contact_multiplier = 1.0, degree_reduction = 'random', npi_risk_multiplier = 1.0, npi_threshold = 500,
                 waning = (180,50), include_vaccination = False, p_vac = (30, 10, 0.8), waning_v = None, \
                 a_seasonality = 0.0, phi_seasonality = 1.0, time_dependent_multiplier = None, label = None, **kwargs):

        print ("=========================================================================")
        print ("Setting up new simulation...")
        
        l2pop = self.l2pop
        npop = self.npop
        sus = self.sus
        tra = self.tra
        homogeneous = self.homogeneous

        m_ij = copy.deepcopy(self.m_ij)

        self.label = label
        
        if risk_multiplier is None:
            print ("Using stored risk multiplier %f." % self.risk_multiplier)
            risk_multiplier = self.risk_multiplier
        else:
            print ("Resetting risk multiplier.")
            self.risk_multiplier = risk_multiplier
    
        print ("Relative amplitude of seasonality: %f" % a_seasonality)    
        seasonality = 1 + a_seasonality * np.cos (2.0*np.pi*np.arange(nsteps)/365.2422 + phi_seasonality)
        if time_dependent_multiplier is not None:
            seasonality *= time_dependent_multiplier
        self.a_seasonality = a_seasonality
        self.phi_seasonality = phi_seasonality
        
        # Initialize seed infections
        self.degree_weighting = degree_weighting
        if degree_weighting:
            print ("Probability of seed infection proportional to number of contacts.")
            jj = self.rng.choice (npop, n_seed, replace=False, p = self.degrees / np.sum(self.degrees))
        else:
            jj = self.rng.choice (npop, n_seed, replace=False)

        print ("Contacts of seed infections:", self.degrees[jj])
        self.t_inf_ini = np.ones(npop) + 9999999
        self.t_inf_ini[jj] = -1
        
        if (intervention == 'contacts') or (intervention == 'both') or (intervention==True):
            print ("Contact reduction due to NPIs by factor %f." % npi_contact_multiplier)
            include_npis = True
        if (intervention == 'risk') or (intervention == 'both') or (intervention==True):
            print ("Risk reduction due to NPIs by factor %f." % npi_risk_multiplier)
            include_npis = True
        else:
            npi_risk_multiplier = 1.0
        if (intervention != 'contacts') and (intervention != 'risk') and (intervention != 'both') and (intervention!=True):
            print ("NPIs switched off.")
            include_npis = False
        

        # All parameters set up, now initialize arrays for simulation
        print ("-------------------------------------------------------------------------")
        print ("Running simulation...")        
        t_inf = np.copy (self.t_inf_ini)
        time = 0
        cinf = np.int64(t_inf<=0)
        cases_ini = np.sum(cinf)
        
        cases_cum = []
        cases = []
        
        block = []
        avg_t = [] # average contacts of infectives
        avg_s = [] # average susceptibility
        avg_r = [] # average risk
        Rt = []

        dc2 = np.zeros((nsteps,l2pop))
        cav = np.zeros((nsteps,l2pop))
        di2 = np.zeros((nsteps,l2pop))
        iav = np.zeros((nsteps,l2pop))
        dci = np.zeros((nsteps,l2pop))

        self.I = []

        p=2**np.arange(l2pop)
    
        activate_npis = False

        t_vac = self.rng.logistic (p_vac[0], p_vac[1], size = npop)
        t_vac += 999999 * (self.rng.uniform (size = npop) > p_vac[2]) # unvaccinated individuals
        self.t_vac = t_vac

        age_inf_max = np.argmax(self.infectivity)
        
        # Simulation loop
        for time in tqdm(range(nsteps)):

            age_inf = time-t_inf
            age_inf = np.maximum (age_inf,0.0)
            inf_current = np.int64(age_inf>0.)
            inf_current_wt = inf_current * tra
            if self.dur_type == 'uniform':
                inf_current_wt *= self.infectivity[np.int64(np.minimum(age_inf,self.tau-1))]
            else:
                dage_inf = age_inf - age_inf_max
                sgn = np.heaviside (dage_inf, 0.0)
                inf_current_wt *= self.infectivity_function(np.maximum (np.minimum( \
                            age_inf_max+(1-sgn)*dage_inf+sgn*dage_inf/self.dur, self.tau-1), 0.0))
            self.I.append(np.sum(inf_current_wt>0.0))

            age_vac = time-t_vac
            vac_current = np.int64(age_vac>=0.)

            if include_npis:
                if not activate_npis:
                    if np.sum(cinf) > npi_threshold:
                        print("Activating NPIs.")
                        if (intervention == 'contacts') or (intervention == 'both') or (intervention==True):
                            if degree_reduction=='random':
                                print ("Random reduction of contacts.")
                                m_ij = self.random_degree_reduction (m_ij, npi_contact_multiplier)
                            elif degree_reduction =='semirandom':
                                print ("Semi-random reduction of contacts (reduced variance).")
                                m_ij = self.semi_random_degree_reduction (m_ij, npi_contact_multiplier)
                            else:
                                raise Exception ("Method for reduction of contacts not specified!")
                            print ("Continuing with reduced number of contacts.")
                            self.degrees_reduced = np.array(np.sum(m_ij.sign(),axis=1))[:,0]
                            print("Average degree now: %f" % np.average(self.degrees_reduced))
                        activate_npis = True
                if activate_npis:
                    inf_current_wt *= npi_risk_multiplier

            risk = seasonality[time] * risk_multiplier * np.array(m_ij @ inf_current_wt)
            
            scr = (risk > 0)
            block = np.append(block, np.sum(scr*(1.0-cinf))/np.sum(scr))
            avg_t = np.append(avg_t, np.sum(scr)/np.sum(inf_current_wt>0))
            avg_s = np.append(avg_s, np.sum(scr*sus)/np.sum(scr))
            avg_r = np.append(avg_r, np.sum(scr*risk)/np.sum(scr))
        
            if homogeneous:
                risk[:] = np.average(risk)        
                
            jj = np.where(risk>0.)[0]
            prob = 1.0 - np.exp(-risk[jj]*sus[jj])
            t_w_eff = age_inf[jj]/self.imm[jj]
            if waning == 'stein':
                # Fit mimicking the data from Stein et al., Lancet  401, 833 (2023)
                prob *= 1 - (0.85-0.35*logistic(315,45).cdf(t_w_eff)) * inf_current[jj]
            elif waning == 'stein_modified':
                prob *= 1 - (1-0.35*logistic(315,45).cdf(t_w_eff)) * inf_current[jj]
            else:
                # tunable logistic function
                prob *= 1 - (1-logistic(waning[0],waning[1]).cdf(t_w_eff)) * inf_current[jj]

            if include_vaccination:
                t_w_eff = age_vac[jj]/self.imm[jj]
                if waning_v == 'stein':
                    # Fit mimicking the data from Stein et al., Lancet  401, 833 (2023)
                    prob *= 1 - (0.85-0.35*logistic(140,30).cdf(t_w_eff)) / (1+(t_w_eff/500)**2) * vac_current[jj]
                elif waning_v == 'stein_modified':
                    prob *= 1 - (1-0.35*logistic(140,30).cdf(t_w_eff)) / (1+(t_w_eff/500)**2) * vac_current[jj]
                else:
                    # tunable logistic function
                    prob *= 1 - (1-logistic(waning_v[0],waning_v[1]).cdf(t_w_eff)) * vac_current[jj]
                    
            infect = self.rng.binomial(1,prob,size=len(jj))
            t_inf[jj] = infect*time + (1-infect)*t_inf[jj]
            n_infect = np.sum(infect)
            
            cases_cum = np.append(cases_cum, len(np.where(t_inf<=time)[0]))
            cases = np.append(cases, n_infect)
            
            # Analyse spectrum
            cinf = np.int64(t_inf<=time)
            inf = np.int64((age_inf>=1.) & (age_inf<=4))
            for kk in range (l2pop):
                c_p = np.reshape(cinf, (npop//2**kk,2**kk))
                i_p = np.reshape(inf, (npop//2**kk,2**kk))
                c_p_av = np.average(c_p,axis=1)
                i_p_av = np.average(i_p,axis=1)
                cav[time,kk] = np.average(c_p_av)
                iav[time,kk] = np.average(i_p_av)
                dc2[time,kk] = np.average(c_p_av**2)-cav[time,kk]**2
                di2[time,kk] = np.average(i_p_av**2)-iav[time,kk]**2
                dci[time,kk] = np.average((c_p_av-cav[time,kk])*(i_p_av-iav[time,kk]))

        print ("-------------------------------------------------------------------------")
        print ("Fraction of population infeceted: %f" % (cases_cum[-1]/npop))

        
        # Estimate secondary attack rate and population-averaged infectivity
        print ("Estimating secondary attack rate among immediate neighbors...")
        print ("(NOTE: Disregarding seasonal modulation for calculation)")
        cum_risk = np.zeros (npop)
        t_inf_tmp = np.zeros (npop)-999999
        # For computing the SAR, must only consider neighbors of nodes that were
        # actualy infected *before* their neighbors
        inf_actual = (t_inf<t_inf[np.bitwise_xor (np.arange(npop),1)])
        for time in tqdm (range (self.tau * np.int64(np.ceil(np.max(self.dur))))):
            age_inf = time
            if self.dur_type == 'uniform':
                inf_current_wt = tra * inf_actual * \
                    self.infectivity_function (np.minimum(np.ceil(age_inf/self.dur),self.tau-1))
            else:
                inf_current_wt = tra * inf_actual * \
                    self.infectivity[np.int64(np.minimum(age_inf,self.tau-1))]
            risk = risk_multiplier * np.array((self.m_ij.multiply(self.neighbors)) @ inf_current_wt)
            scr = (risk > 0)
            if homogeneous:
                risk[:] = np.average(risk)
            jj = np.where(risk>0.)[0]
            cum_risk += risk*sus
            prob = 1.0 - np.exp(-risk[jj]*sus[jj])
            infect = self.rng.binomial(1,prob,size=len(jj))
            t_inf_tmp[jj] = infect*time + (1-infect)*t_inf_tmp[jj]
        
        print("Secondary attack rate:")
        self.sar = np.sum(1.0-np.exp(-cum_risk))/np.sum(inf_actual)
        print("%f (analytical)" % self.sar)
        print("%f (Monte Carlo simulation)" % (np.sum(t_inf_tmp>=0)/np.sum(inf_actual)))

        # Estimate overdispersion
        print ("Estimating overdispersion of secondary cases...")
        m_test = 0 * self.m_ij
        for time in tqdm (range (self.tau * np.int64(np.ceil(np.max(self.dur))))):
            age_inf = time
            if self.dur_type == 'uniform':
                inf_current_wt = tra * inf_actual * \
                    self.infectivity_function (np.minimum(np.ceil(age_inf/self.dur),self.tau-1))
            else:
                inf_current_wt = tra * inf_actual * \
                    self.infectivity[np.int64(np.minimum(age_inf,self.tau-1))]
            m_test += sparse.diags_array(self.sus).tocsr() @ (self.m_ij @ sparse.diags_array(inf_current_wt).tocsr())

        m_test = -m_test*self.risk_multiplier
        m_test = -m_test.expm1()
        m_test.data = self.rng.binomial(1,m_test.data)

        sec = np.ndarray.flatten (np.array (np.sum (m_test,axis=0)))
        
        jj = np.where(t_inf<=nsteps)
        self.sec = sec[jj[0]]
        self.mu = np.average(self.sec)
        self.s2 = np.var(self.sec)
        self.k_ovd = self.mu**2/(self.s2-self.mu)
        self.sec_hist = np.histogram (self.sec, bins=np.arange (np.max (self.sec)+1))
        print("Average number of secondary infection mu = %f" % self.mu)
        print("Overdispersion k = %f" % self.k_ovd)


        # Wrap-up: Store results as object attributes and compute some auxiliary diagnostics       
        self.I = np.array(self.I)
        self.C = np.array(cases_cum/npop)
        self.cases = np.array(cases)

        # Estimate Rt using method of Cori et al.
        cases0 = np.append([cases_ini], self.cases)
        nmax = min(len(self.infectivity_avg)*2,nsteps-2)//2
        tmp = np.append(0*self.infectivity_avg[1:nmax], self.infectivity_avg[:nmax])
        tmp = tmp / np.sum(tmp)
        self.Rt = self.cases / np.convolve (cases0, tmp, mode ='same')[1:]
           
        self.block = block
        self.avg_r = avg_r
        self.avg_s = avg_s
        self.avg_t = avg_t
        self.dc2 = dc2
        self.cav = cav
        self.di2 = di2
        self.iav = iav
        self.dci = dci
        self.p = p
        self.t_inf = t_inf
        self.risk_multiplier = risk_multiplier

    print ("\n")
    print ("=========================================================================")
        
        
    def tune_rt (self, nsteps = 50, r0c = 3.0, tol = 0.1, calibration_point = 'infections', \
                 c_cal = 2e-3, i_start = 5, i_end = 10, nmax = 3, \
                 risk_multiplier = 0.12, intervention = False, \
                 waning = (180,50), alpha = 0.5, a_seasonality = 0.0, phi_seasonality = 1.0, **kwargs):
        
        self.risk_multiplier = risk_multiplier
        r0_test_old = 1e33
        risk_multiplier_old = self.risk_multiplier
        
        for iter in range(20):
            self.simulate (nsteps = nsteps, risk_multiplier = self.risk_multiplier, intervention = intervention, \
                           waning = waning, a_seasonality = a_seasonality, phi_seasonality = phi_seasonality, **kwargs)
            if calibration_point == 'maximum':
                    r0_test = np.sort (np.log(self.Rt))
                    jj = np.where(~(np.isnan(r0_test) | np.isinf(r0_test)))[0]
                    r0_test = np.exp (np.average (r0_test[jj[-nmax:]]))
            elif calibration_point == 'time':
                    r0_test = np.exp (np.average (np.log (self.Rt[i_start:i_end+1])))
                    print (self.Rt[i_start:i_end+1])
            elif calibration_point == 'infections':
                ii1 = np.where((self.C < c_cal) & (self.cases>0))[0][-1]
                ii0 = np.max([ii1-10,0])
                r0_test = np.exp (np.average (np.log (self.Rt[ii0:ii1+1])))
            else:
                raise Exception ("Wrong choice for parameter 'calibration_point'.")
            #r0_test = (np.sum (self.cases[i_start+igen*4:i_start+igen*4+4]) / \
            #           np.sum (self.cases[i_start:i_start+4]))**(1.0/igen)
            r0_test = max (r0_test, 0.1)
            print("Risk multiplier=%f, R0=%f" % (self.risk_multiplier, r0_test))
            if np.abs (np.log (r0_test) / np.log(r0c) - 1.0) < tol:
                break

            if (np.log (r0_test/r0c)*np.log (r0_test_old/r0c) < 0.0) and r0_test_old<1e18:
                self.risk_multiplier = 0.5 * (self.risk_multiplier + risk_multiplier_old)
            else:
                self.risk_multiplier *= (r0c/r0_test) ** alpha

            if self.C[i_end+1]>0.3:
                self.risk_multiplier = risk_multiplier_old * 0.95 
               
            risk_multiplier_old = self.risk_multiplier


    def clear (self, n_contacts_l = True, m_ij = True, degrees = True, neighbors = True, dur = True, \
               imm = True, tra = True, sec = True, sec_hist = False, sus = True, t_inf = True, t_inf_ini = True):

        if n_contacts_l: self.n_contacts_l = None
        if m_ij: self.m_ij = None
        if degrees: self.degrees = None
        if neighbors: self.neighbors = None
        if dur: self.dur = None
        if imm: self.imm = None
        if tra: self.tra = None
        if sus: self.sus = None
        if t_inf_ini: self.t_inf_ini = None
        if t_inf: self.t_inf = None
        if sec: self.sec = None
        if sec_hist: self.sec_hist = None

    ########################################################################################
    
    def random_degree_reduction (m_ij, frac0):
        tmp = sparse.triu (m_ij.tocoo())
        val = tmp.data
        coords = tmp.coords
        jj = self.rng.choice (len(val), size=np.int64(frac0*len(val)), replace=False)
        m_ij = sparse.coo_matrix ((val[jj], (coords[0][jj], coords[1][jj])), shape = m_ij.shape).tocsr()
        m_ij = sparse.spmatrix.maximum (m_ij, m_ij.transpose()).tocsr()
        return m_ij


    def semi_random_degree_reduction (m_ij, frac0):
        tmp = sparse.triu(m_ij, format='csr')
        indptr_old = tmp.indptr
        indices_old = tmp.indices
        data_old = tmp.data
        indptr_new = [0]
        indices_new = []
        data_new = []
        n_contacts_total = len(tmp.data)
        ncnew = 0
        ncold = 0
        frac=frac0
        for row in range(len(indptr_old)-1):
            nscr = indptr_old[row+1]-indptr_old[row] 
            new = self.rng.choice(indices_old[indptr_old[row]:indptr_old[row+1]],np.int64(np.round(nscr*frac)),replace=False)
            ncnew += len(new)
            ncold += nscr
            indices_new += new.tolist()                                   
            data_new += data_old[new].tolist()
            indptr_new.append (ncnew)
            frac = (n_contacts_total*frac0 - ncnew) / (n_contacts_total - ncold + 1e-30)
        data_new = np.array(data_new)
        indices_new = np.array(indices_new)
        indptr_new = np.array(indptr_new)
        m_ij = sparse.csr_matrix ((data_new, indices_new, indptr_new), shape = m_ij.shape)
        m_ij = sparse.spmatrix.maximum (m_ij, m_ij.transpose()).tocsr()
        return m_ij


def plot_models (models, labels):
    
    for m, l in zip(models,labels):
        plt.plot(m.Rt,linewidth=0.6,label=l)
    plt.legend()
    plt.ylim(0.0,5)
    plt.axhline(1,color='k',linestyle='dotted')
    plt.xlabel(r'Time (days)')
    plt.ylabel(r'$\mathcal{R}(t)$')
    plt.xlim(0,200)
    plt.show()

    for m, l in zip(models,labels):
        plt.plot(m.C,m.Rt,linewidth=0.6,label=l)
    plt.legend()
    plt.axhline(1,color='k',linestyle='dotted')
    plt.xscale('log')
    plt.xlabel(r'$\bar{C}$')
    plt.ylabel(r'$\mathcal{R}(t)$')
    plt.ylim(0,5)
    plt.show()

    for m, l in zip(models,labels):
        plt.plot(m.C,linewidth=0.6,label=l)
    plt.legend()
    plt.xlabel(r'Time (days)')
    plt.ylabel(r'$\bar{C}(t)$')
    plt.yscale('log')
    plt.show()

    for m, l in zip(models,labels):
        plt.plot(m.C,linewidth=0.6,label=l)
    plt.legend()
    plt.xlabel(r'Time (days)')
    plt.ylabel(r'Cumulative case fraction')
    plt.show()

    for m, l in zip(models,labels):
        plt.plot(m.cases,linewidth=0.6,label=l)
    plt.legend()
    plt.xlabel(r'Time (days)')
    plt.ylabel(r'Cases per day')
    plt.yscale('log')


def plot_spectra (data, ii, ilast, models, variable=None, reporting_factor = 3, norm = 'absolute', \
                  show_dC2 = True, show_dCdI = True, show_dI2 = True, incr = np.sqrt(2), label = None, \
                  fontsize = 10, ncol = 1):
    
    # Show dC^2/C^2
    xtmp = 10**np.arange(0,8,0.2)
    legend_handles = []
    
    if norm == 'absolute':
        print ("Matching absolute fraction of cumulative infections for data and simulations.")
        crit = reporting_factor*data.Cav[ii,-1]
    elif norm == 'relative':
        print ("Matching fraction of cumulative infections relative to outbreak size for data and simulations.")
        crit = data.Cav[ii,-1]/data.Cav[ilast,-1]
    else:
        raise Exception ("Wrong choice for parameter 'norm'.")
    
    if show_dC2:
        qc1 = plt.scatter(data.p[:],data.dC2[ii,:]/(data.Cav[ii,:]*data.Cav[ii,:]),marker='+',s=25,color='b',\
                    label=r'$\langle\delta C^2\rangle/\bar{C}^2$')
        legend_handles.append(qc1)
    if show_dCdI:
        qc2 = plt.scatter(data.p[:],data.dCdI[ii,:]/(data.Iav[ii,:]*data.Cav[ii,:]),marker='X',s=25,color='r',\
                    label=r'$\langle\delta C\, \delta I\rangle/(\bar{C} \bar{I}$)')
        legend_handles.append(qc2)
    if show_dI2:
        qc3 = plt.scatter(data.p[:],data.dI2[ii,:]/(data.Iav[ii,:]*data.Iav[ii,:]),marker='o',s=25,color='orange',\
                    label=r'$\langle\delta I^2\rangle/\bar{I}^2$')
        legend_handles.append(qc3)
        
    qc4, = plt.plot(xtmp,np.exp(data.alpha1[ii]+np.log(xtmp)*data.alpha0[ii])/data.Cav[ii,-1]**1,c='b',linestyle='dashed',label='fit')
    legend_handles.append(qc4)
    Ctmp = reporting_factor*data.Cav[ii,-1]
    plt.fill_between(xtmp,xtmp**0/Ctmp,\
                     5*xtmp**0/Ctmp,color='grey',alpha=0.3)
    plt.fill_between(xtmp,xtmp**0*1e-12,(1-Ctmp)/(xtmp*Ctmp),color='grey',alpha=0.3)
      
    plt.axvline(1.0/Ctmp,linestyle='dotted')
    plt.axhline(1.0,linestyle='dotted')
    
    plt.yscale('log')
    plt.xscale('log')
    
    xt_major = 10**np.arange(8)
    yt_major = 10**np.arange(-3.,np.log10(4/(reporting_factor*data.Cav[ii,-1])))
      
    plt.xticks (xt_major)
    plt.yticks (yt_major)
    plt.xlabel (r'Scale $p$')
    plt.ylabel (r'Normalized correlation')
    plt.minorticks_on()
    plt.text(5,5e-3, pd.to_datetime(data.date[ii], unit='D', origin='julian').strftime('%d %b %Y'))
    plt.ylim(1e-3,4/(reporting_factor*data.Cav[ii,-1]))
    plt.xlim(1,5e7)
    plt.xlabel(r'Scale $p$')
    plt.ylabel(r'Correlation function')
    
    #Simulated spectra 
    ithi=0.5
    for m in models:        
        if norm == 'absolute':
            try:
                print ("Matching outbreak size %f." % crit)
                ii=np.where(m.cav[:,-1]>=crit)[0][0]
            except:
                continue
        elif norm == 'relative':
            print ("Matching fraction %f of outbreak size." % crit)
            ii=np.where(m.cav[:,-1]>=crit*m.cav[-1,-1])[0][0]

        if show_dC2:
            plt.plot(m.p,m.dc2[ii,:]/m.cav[ii,:]**2,linewidth=ithi,color='b')
        if show_dCdI:
            plt.plot(m.p,m.dci[ii,:]/(m.cav[ii,:]*m.iav[ii,:]),linewidth=ithi,color='r')
        if show_dI2:
            plt.plot(m.p,m.di2[ii,:]/m.iav[ii,:]**2,linewidth=ithi,color='orange')
        if norm == 'absolute':
            print("%f of outbreak size" % (crit/m.C[-1]))
        elif norm == 'relative':
            print("Current outbreak size in model: %f" % (crit*m.C[-1]))

        if variable is not None:
            if variable[0]==r'\theta':
                qc, = plt.plot ([-1e99,-2e99], [0,0], color='k', linewidth=ithi, label=r'$%s = %g$' % (variable[0], m.p_sus[1]))
            elif variable[0]==r'k_\mathrm{tra}' and vars(m)[variable[1]]==None:
                qc, = plt.plot ([-1e99,-2e99], [0,0], color='k', linewidth=ithi, label=r'Uniform')                
            else:
                qc, = plt.plot ([-1e99,-2e99], [0,0], color='k', linewidth=ithi, label=r'$%s = %g$' % (variable[0], vars(m)[variable[1]]))
            legend_handles.append(qc)
            
        ithi=incr*ithi

    plt.legend (handles = legend_handles,fontsize=fontsize,loc='upper right',ncol=ncol)

    if label == None:
        plt.savefig('spectrum.pdf')
    else:
        plt.annotate('Series '+label,(2/Ctmp,1.5/Ctmp))
        plt.savefig('spectrum_'+label+'.pdf')
    plt.show()


def plot_spec_fixed_scale (data, ilast, models, variable=None, p_sim = -3, incr = np.sqrt(2), label=None, \
                           show_dC2 = True, show_dCdI = True, show_dI2 = True, logx = False, logy=True, \
                           fontsize = 10, ncol=1):

    legend_handles = []
    if show_dC2:
        plt.plot(data.Cav[:,-1]/data.Cav[ilast,-1],data.dC2[:,0]/data.Cav[:,0]**2,color='b',linestyle='dashed', \
              label=r'$\langle\delta C^2\rangle/\bar{C}^2$')
        qc1, = plt.plot([-2e99,-1e99],[0,0],color='b',label=r'$\langle\delta C^2\rangle/\bar{C}^2$')
        legend_handles.append(qc1)
    if show_dCdI:
        plt.plot(data.Cav[:,-1]/data.Cav[ilast,-1],data.dCdI[:,0]/data.Cav[:,0]/data.Iav[:,0],color='r',linestyle='dashed', \
              label=r'$\langle\delta C\,\delta I\rangle/(\bar{C}\bar{I})$')
        qc2, = plt.plot([-2e99,-1e99],[0,0],color='r',label=r'$\langle\delta C\,\delta I\rangle/(\bar{C}\bar{I})$')
        legend_handles.append(qc2)
    if show_dI2:
        plt.plot(data.Cav[:,-1]/data.Cav[ilast,-1],data.dI2[:,0]/data.Iav[:,0]**2,color='orange',linestyle='dashed',\
              label=r'$\langle\delta I^2\rangle/\bar{I}^2$')
        qc3, = plt.plot([-2e99,-1e99],[0,0],color='orange',label=r'$\langle\delta I^2\rangle/\bar{I}^2$')
        legend_handles.append(qc3)
    ithi = 0.5
    
    for m in models:
        print ("Showing model spectrum at scale p=%i." % m.p[p_sim])
        if show_dC2:
            plt.plot(m.C/m.C[-1],m.dc2[:,p_sim]/m.cav[:,p_sim]**2,color='b',linewidth=ithi)
        if show_dCdI:
            plt.plot(m.C/m.C[-1],m.dci[:,p_sim]/m.cav[:,p_sim]/m.iav[:,p_sim],color='r',linewidth=ithi)
        if show_dI2:
            plt.plot(m.C/m.C[-1],m.di2[:,p_sim]/m.iav[:,p_sim]**2,color='orange',linewidth=ithi)
        if variable is not None:
            if variable[0]==r'\theta':
                qc, = plt.plot ([-1e99,-2e99], [0,0], color='k', linewidth=ithi, label=r'$%s = %g$' % (variable[0], m.p_sus[1]))
            elif variable[0]==r'k_\mathrm{tra}' and vars(m)[variable[1]]==None:
                qc, = plt.plot ([-1e99,-2e99], [0,0], color='k', linewidth=ithi, label=r'Uniform')                
            else:
                qc, = plt.plot ([-1e99,-2e99], [0,0], color='k', linewidth=ithi, label=r'$%s = %g$' % (variable[0], vars(m)[variable[1]]))
            legend_handles.append(qc)
        ithi = incr*ithi
    
    if logx:
        plt.xscale('log')
        plt.xlim(1e-5,1)
    else:
        plt.xlim(0,1)
    if logy: 
        plt.yscale('log')
        plt.ylim(1e-2,10)

    plt.ylabel(r'Correlation functions at scale $p=2^{%i}$' % p_sim)
    plt.xlabel(r'$\bar{C}/\bar{C}(\infty)$')
    plt.minorticks_on()
    plt.legend (handles = legend_handles, loc='upper right', fontsize=fontsize, ncol=ncol)
    
    if label is not None:
        plt.annotate('Series '+label,(0.12,25))
        
    
def autocorrelation (model, index, window = 20, normalize = True):

    # WARNING: This function assumes that we have no reinfections!
    
    i0av = np.zeros((2*window+1,model.l2pop))
    i1av = np.zeros((2*window+1,model.l2pop))
    didi = np.zeros((2*window+1,model.l2pop))

    inf0 = model.t_inf==index
        
    for ii in range(index-window,index+window+1):
        inf1 = model.t_inf==ii
        time = ii - (index-window)
        for kk in range (model.l2pop):
            i0_p = np.reshape(inf0, (model.npop//2**kk,2**kk))
            i0_p_av = np.average(i0_p,axis=1)
            i1_p = np.reshape(inf1, (model.npop//2**kk,2**kk))
            i1_p_av = np.average(i1_p,axis=1)
            i0av[time,kk] = np.average(i0_p_av)
            i1av[time,kk] = np.average(i1_p_av)
            didi[time,kk] = np.average((i0_p_av-i0av[time,kk])*(i1_p_av-i1av[time,kk])) / \
                (i0av[time,kk]*i1av[time,kk])

    if normalize:
        didi[:,:] = didi[:,:] / (didi[window,:])[np.newaxis,:]
            
    return {'didi' : didi, 'i1av' : i1av, 'i0av' : i0av, 'time' : np.arange(-window,window+1)}



def autocorrelation_observed (cases, pop, index, window = 30, normalize = True):

    I = cases / pop[:,np.newaxis]
    I0 = I[:,index]
    Iav0 = np.average (I0, weights=pop)
    dI0 = I0 -Iav0

    dIdI = np.zeros(2*window+1)
    Iav1 = np.zeros(2*window+1)

    for ii in range(index-window,index+window+1):
        I1 = I[:,ii]
        Iav1[ii-index+window] = np.average (I1, weights=pop)
        dI1 = I1 -Iav1[ii-index+window]
        dIdI[ii-index+window] = np.average(dI1*dI0, weights=pop) / \
            (Iav0 * Iav1[ii-index+window])

    if normalize:
        dIdI = dIdI / dIdI[window]

    return {'dIdI' : dIdI, 'Iav0' : Iav0, 'Iav1' : Iav1, 'time' : np.arange(-window,window+1)}



def autocorrelation_observed2 (cases, pop):

    C = np.cumsum (cases,axis=1) / pop[:,np.newaxis]
    I = cases / pop[:,np.newaxis]
    Cav = np.average (C, weights=pop, axis=0)
    Iav = np.average (I, weights=pop, axis=0)
    dC = C - Cav [np.newaxis,:]
    dC = C / Cav[np.newaxis,:] * pop[:,np.newaxis]
    dI = I - Iav [np.newaxis,:]
    dI = I / Iav[np.newaxis,:] * pop[:,np.newaxis]

    nn = len(C[0,:])

    dC2 = np.sqrt (np.sum (dC**2,axis=0))
    dI2 = np.sqrt (np.sum (dI**2,axis=0))
     
    dCdC = np.tensordot (dC, dC, axes=(0,0)) / dC2[:,np.newaxis] / dC2[np.newaxis,:]
    dCdI = np.tensordot (dC, dI, axes=(0,0)) / dC2[:,np.newaxis] / dI2[np.newaxis,:]
    dIdI = np.tensordot (dI, dI, axes=(0,0)) / dI2[:,np.newaxis] / dI2[np.newaxis,:]

    return {'dCdC' : dCdC, 'dCdI' : dCdI, 'dIdI' : dIdI}


def summary(series, variable, label):
    print("%7s %7s %7s %7s %7s %7s %7s" % (label, r'R_C', r'C(∞)', r'C(∞)/C0', 'SAR', 'k', 'av.deg.'))
    for m in series:
        jj = np.where((m.C<5e-3) & (m.C>5e-4))[0]
        rtmp = np.exp(np.mean(np.log(m.Rt[jj])))
        r_inf = 1.0+lambertw(-np.exp(-rtmp)*rtmp)/rtmp
        param = vars(m)[variable]
        try:
            if len(param)>0:
                param=param[0]
        except:
            param=param
        if param is None:
            param = np.nan
        print("%7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f" % \
              (param, rtmp, m.C[-1], np.real(m.C[-1]/r_inf), m.sar, m.k_ovd, m.avg_degree))

def summary_latex(series, label, variable, header = False):
    
    if header:
        print(" & %7s & %7s & %7s & %7s & %7s & %7s & %7s & %7s & %7s & %7s & %7s & %7s & %7s \\\\" % \
              ('$f_\\mathrm{c}$', '$\\gamma_\\mathrm{c}$', '$\\gamma_\\mathrm{r}$', \
               '$\\delta$', '$\\theta$', '$\\alpha_\\mathrm{P}$', \
               '$k_\\mathrm{tra}$', 'avg.\\ deg.', '$R_\\mathrm{C}$', \
               '$C(\\infty)$', '$C(\\infty)/C_0$', 'SAR', '$k_\\mathrm{ovd}$'))
        print("\\hline")
    
    #print("\\multirow{%i}{*}{\\rotatebox[origin=c]{90}{%s}}" % (len(series), label))
    for m in series:
        jj = np.where((m.C<5e-3) & (m.C>5e-4))[0]
        rtmp = np.exp(np.mean(np.log(m.Rt[jj])))
        r_inf = 1.0+lambertw(-np.exp(-rtmp)*rtmp)/rtmp
        if label == "CgR":
            rtmp = np.nan
        
        param = vars(m)[variable]
        try:
            if label == "CSI":
                model_string = "%s-%g" % (label, m.p_sus[1])
            elif label == "CSIDT":
                if m.k_tra is None:
                    param = 0
                else:
                    param = 1/m.k_tra
                model_string = "%s-%g" % (label, param)
            else:
                model_string = "%s-%3.2f" % (label, param)
        except:
            model_string = "%s" % label
        
        if m.sus_type == 'uniform':
            scr1 = np.nan
        else:
            scr1 = m.p_sus[1]
        if m.degree_dist == 'pareto':
            scr2 = m.alpha_pareto
        else:
            scr2 = np.nan
            
        try:
            scr3 = 1/m.k_tra
            k_tra_str = "1/%i" % scr3
        except:
            k_tra_str = "---"
        
        outstr = " %s & %7.2f & %7.2f & %7.2f & %7.2f & %7.2f & %7.2f & %s & %7.2f & %7.2f & %7.2f & %7.2f & %7.2f & %7.2f \\\\" % \
              (model_string, m.f_cluster, m.gamma_c, m.gamma_r, m.p_scal, scr1, scr2, k_tra_str, \
               m.avg_degree, rtmp, m.C[-1], np.real(m.C[-1]/r_inf), m.sar, m.k_ovd)
        print(outstr.replace("nan","---"))
    print("\\hline")
