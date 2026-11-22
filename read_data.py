import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import pandas as pd
import glob
from tqdm import tqdm
import scipy.stats as stats
import statsmodels.api as sm
import copy
import os
import requests
import sys
import pickle
from zipfile import ZipFile


class case_data:

    def __init__ (self):
    
        # read files (and download if needed)
        file = 'data/Aktuell_Deutschland_SarsCov2_Infektionen.csv'
        if glob.glob (file) == []:
            print ("Downloading data...")
            print ("In case of failure, please download https://zenodo.org/records/21484070/files/Aktuell_Deutschland_SarsCov2_Infektionen.csv")
            download ("https://zenodo.org/records/21484070/files/Aktuell_Deutschland_SarsCov2_Infektionen.csv")
        rki_cases = pd.read_csv(file).sort_values (by = 'Meldedatum') 

        file = 'data/COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv'
        if glob.glob (file) == []:
            print ("Downloading data...")
            print ("In case of failure, please download https://zenodo.org/records/21484070/files/Aktuell_Deutschland_SarsCov2_Infektionen.csv")
            download ("https://zenodo.org/records/22742425/files/COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv")
        lk_list = pd.read_csv(file)
        
        lkID = sorted(rki_cases.IdLandkreis.unique())

        nc = len(lkID)
        print ('%i Landkreise' % nc)
        pop = np.zeros(nc)
        lk = []

        # Create arrays of counties and assign population
        for ic in range(nc):
            pop[ic] = lk_list['Bevoelkerung'][lk_list['Landkreis_id']==lkID[ic]].unique()
        lkID = np.array (lkID)

        
        Meldedatum = pd.to_datetime (rki_cases.Meldedatum.to_numpy()).to_julian_date()
        Refdatum   = pd.to_datetime (rki_cases.Refdatum.to_numpy()).to_julian_date()
        AnzahlFall = rki_cases.AnzahlFall.to_numpy()
        Erkrankung = rki_cases.IstErkrankungsbeginn.to_numpy()
        
        date_rki = np.arange (np.min(Meldedatum), np.max(Meldedatum)+0.5)
        nt_rki = len(date_rki)
        
        age_groups = rki_cases.Altersgruppe.unique()[:-1]
        n_age = len(age_groups)
        
        Altersgruppe = rki_cases.Altersgruppe.to_numpy()
        Geschlecht = rki_cases.Geschlecht.to_numpy()
        
        delay = Meldedatum - Refdatum
        delay_max = 28
        p_delay  = np.zeros((nt_rki, delay_max, n_age, 2))
        n_groups = np.zeros((nt_rki, n_age, 2))
        delay_av = np.zeros((nt_rki, n_age, 2))
        
        mask_age = np.zeros((len(Meldedatum), n_age))
        for i_a in range(n_age):
            mask_age [:,i_a] = (Altersgruppe==age_groups[i_a]).astype(np.int32)
        
        case_sym = []
        delay_groups = []
        date_groups = []
        
        for i_a in range (n_age):
            jjm = np.where ((Altersgruppe==age_groups[i_a]) & (Geschlecht=='M') & (delay>=0) & (delay<28))
            jjf = np.where ((Altersgruppe==age_groups[i_a]) & (Geschlecht=='W') & (delay>=0) & (delay<28))
            case_sym.append ([(AnzahlFall*Erkrankung)[jjm[0]], (AnzahlFall*Erkrankung)[jjf[0]]])
            delay_groups.append ([delay[jjm[0]], delay[jjf[0]]])
            date_groups.append ([Meldedatum[jjm[0]], Meldedatum[jjf[0]]])        

        # Construct time-dependent delay distribution between symptom onset and reporting date:
        width = 3.5
        bins = np.arange(0,delay_max+1) - 0.5
        binc = np.arange(0,delay_max)
        
        for ii in tqdm(range(nt_rki)):
            for i_a in range(n_age):
                win = stats.norm.pdf (date_groups[i_a][0] - date_rki[ii], scale=width)
                n_groups[ii,i_a,0] = np.sum (win*case_sym[i_a][0])
                if n_groups[ii,i_a,0]>0:
                    p_delay[ii,:,i_a,0] = \
                        np.histogram (delay_groups[i_a][0], weights = win * case_sym[i_a][0],\
                                     bins=bins, density=True)[0]
                    delay_av[ii,i_a,0] = np.average (binc, weights=p_delay[ii,:,i_a,0])
                
                win = stats.norm.pdf (date_groups[i_a][1] - date_rki[ii], scale=width)
                n_groups[ii,i_a,1] = np.sum (win*case_sym[i_a][1])
                if n_groups[ii,i_a,1]>0:
                    p_delay[ii,:,i_a,1] = \
                        np.histogram (delay_groups[i_a][1], weights = win * case_sym[i_a][1],\
                                     bins=bins, density=True)[0]
                    delay_av[ii,i_a,1] = np.average (binc, weights=p_delay[ii,:,i_a,1])
        
        cp_delay = np.cumsum(p_delay, axis=1)        

        # Stochastically correct for reporting delay and incubation period
        ncase_sym = np.zeros ((nc,nt_rki))
        
        age_group = np.zeros (len (Meldedatum), np.int32)
        for i_a in range(n_age):
            jj = np.where(Altersgruppe == age_group[i_a])
            age_group [jj[0]] = i_a

        sex = np.zeros (len (Meldedatum), np.int32)
        j_male = np.where(Geschlecht == 'M')[0]
        sex [j_male] = 0
        j_female = np.where(Geschlecht == 'W')[0]
        sex [j_female] = 1
        j_unknown = np.where(Geschlecht == 'unbekannt')[0]
        sex [j_unknown] = np.heaviside (np.random.rand(len(j_unknown)) - \
                                        len(j_male)/(len(j_male)+len(j_female)), 0)
        
        # Parameters of gamma-distributed incubation period for wild type from
        # Galmiche et al., Lancet Microbe 2023, https://doi.org/10.1016/S2666-5247(23)00005-8.
        gamma_mean = 4.07
        gamma_sigma = 2.12
        gamma_shape = gamma_mean**2/gamma_sigma**2
        gamma_scale = gamma_sigma**2/gamma_mean
        print ("Using gamma distribution with scale parameter %f and shape parameter %f for incubation period." % (gamma_scale,gamma_shape))
        
        for ic in tqdm(range(nc)):
            jc = np.where (rki_cases.IdLandkreis==lkID[ic])[0]
        
            for ii in jc:
                if Erkrankung[ii]:
                    for ij in range(AnzahlFall[ii]):
                        kk = int(Refdatum[ii]-date_rki[0])
                        kk -= int(np.rint(np.random.gamma(gamma_shape, scale = gamma_scale)))
                        if kk>=0:
                            ncase_sym [ic,kk] += 1
                else:
                    for ij in range(AnzahlFall[ii]):
                        kk = int (Meldedatum[ii]-date_rki[0])
                        i_delay = np.where(cp_delay [kk,:, age_group[ii], sex[ii]] >
                                           np.random.rand())[0][0]
                        kk -= i_delay + int(np.rint(np.random.gamma(gamma_shape, scale = gamma_scale)))
                        if kk>=0:
                            ncase_sym [ic,kk] +=1
            
        # Favre averages and fluctuations        
        i_s = 0
        i_e = np.int64(date_rki[-1]-date_rki[0]) 
        nt = i_e-i_s+1
        ncase = ncase_sym[:,i_s:i_e+1]
        
        pop0=np.tensordot(pop,np.ones(nt),axes=0)
        
        I_act = np.zeros((nc,nt))
        R_act = np.zeros((nc,nt))
        R_act = np.zeros((nc,nt))
        for i in range(nt):
            i0=max(0,i-3)
            I_act[:,i] = np.sum(ncase[:,i0:i+1],axis=1)/pop
        for i in range(4,nt):
            R_act[:,i] = R_act[:,i-1]+ncase[:,i-4]/pop-R_act[:,i-1]/120.
            
        Iav=np.average(I_act,weights=pop0,axis=0)
        Rav=np.average(R_act,weights=pop0,axis=0)
        dI = I_act - np.tensordot(np.ones(nc),Iav,axes=0)
        dR = R_act - np.tensordot(np.ones(nc),Rav,axes=0)
        dIdR = np.average(dI*dR,weights=pop0,axis=0)
        dI2 = np.average(dI*dI,weights=pop0,axis=0)
        dR2 = np.average(dR*dR,weights=pop0,axis=0)

        # Construct scale-dependent Favre decomposition    
        if glob.glob('data/georef-germany-kreis.csv') == []:
            print ("Downloading geospatial data...")
            print ("In case of failure, please download from https://public.opendatasoft.com/explore/dataset/georef-germany-kreis/export/?disjunctive.lan_code&disjunctive.lan_name&disjunctive.krs_code&disjunctive.krs_name&disjunctive.krs_name_short&sort=year&flg=en-us&location=6,51.32946,10.45403&basemap=jawg.light")
            download ("https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-germany-kreis/exports/csv?lang=en&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B", \
                      filename ='georef-germany-kreis.csv')
        dat = pd.read_csv('data/georef-germany-kreis.csv',sep=';')
        lat = np.zeros(nc) 
        lon = np.zeros(nc)

        coords_berlin = {
            1: (52.5373,13.3603),
            2: (52.4996,13.4314),
            3: (52.5929,13.4317),
            4: (52.5053,13.2600),
            5: (52.5352,13.2003),
            6: (52.4309,13.1927),
            7: (52.4722,13.3703),
            8: (52.4811497,13.4353501),
            9: (52.4204,13.6200),
            10: (52.5229,13.5766),
            11: (52.5336,13.4999),
            12: (52.5790,13.2805)}
        
        for ic in range(nc):
        
            try:
                coord = np.float64(dat['Geo Point'][dat['Kreis code']==lkID[ic]].iloc[0].split(','))
                lat[ic] = coord[0]
                lon[ic] = coord[1]
            except:
                print('No coordinates available for Berlin Stadtbezirk %i.' % lkID[ic])
                lat[ic], lon[ic] = coords_berlin[lkID[ic]-11000]
                print('Coordinates set manually.')
        
        from sklearn.metrics.pairwise import haversine_distances
        p = 2e5 * 1.5**np.arange(1,14+1)
        nk=14
        
        coords=[[lat[ic]*np.pi/180,lon[ic]*np.pi/180] for ic in range(nc)]
        dist = haversine_distances(coords)
        ncase_p=np.zeros((nc,nt_rki,nk))
        pop_p=np.zeros((nc,nt_rki,nk))
        pop0=np.tensordot(pop,np.ones(nt_rki),axes=0)
        
        
        ic = 234
        ip = 0
        
        I_act = np.zeros((nc,nt_rki,nk))
        R_act = np.zeros((nc,nt_rki,nk))
        C_act = np.zeros((nc,nt_rki,nk))
        
        Iav = np.zeros((nt_rki,nk))
        Rav = np.zeros((nt_rki,nk))
        Cav = np.zeros((nt_rki,nk))
        dI = np.zeros((nc,nt_rki,nk))
        dR = np.zeros((nc,nt_rki,nk))
        dC = np.zeros((nc,nt_rki,nk))
        dIdR = np.zeros((nt_rki,nk))
        dCdI = np.zeros((nt_rki,nk))
        IR = np.zeros((nt_rki,nk))
        CI = np.zeros((nt_rki,nk))
        dI2 = np.zeros((nt_rki,nk))
        dR2 = np.zeros((nt_rki,nk))
        R2 = np.zeros((nt_rki,nk))
        dC2 = np.zeros((nt_rki,nk))
        C2 = np.zeros((nt_rki,nk))
        Cmax = np.zeros((nt_rki,nk))
        
        for ip in range(nk):
            for ic in range(nc):
                ii = np.argsort(dist[ic,:])
                scr1 = np.cumsum(pop[ii])
                if (max(scr1)<p[ip]):
                    iip = nc-1
                else:
                    iip = np.where(scr1>p[ip])[0][0] #iip closest county exceeding poulation p[ip]
                for jj in ii[:iip+1]:
                    ncase_p[ic,:,ip] += ncase_sym[jj,:]
                    pop_p[ic,:,ip]   += pop0[jj]
           
            #Compute correlations
            for i in range(nt_rki):
                i0=max(0,i-3)
                I_act[:,i,ip] = np.sum(ncase_p[:,i0:i+1,ip],axis=1)/pop_p[:,0,ip]
            for i in range(4,nt_rki):
                R_act[:,i,ip] = R_act[:,i-1,ip]+ncase_p[:,i-4,ip]/pop_p[:,0,ip]#-R_act[:,i-1,ip]/180.
    
            C_act[:,:,ip] = I_act[:,:,ip] + R_act[:,:,ip]
                
            Iav[:,ip]=np.average(I_act[:,:,ip],weights=pop0,axis=0)
            Rav[:,ip]=np.average(R_act[:,:,ip],weights=pop0,axis=0)
            Cav[:,ip]=np.average(C_act[:,:,ip],weights=pop0,axis=0)
            dI[:,:,ip] = I_act[:,:,ip] - np.tensordot(np.ones(nc),Iav[:,ip],axes=0)
            dR[:,:,ip] = R_act[:,:,ip] - np.tensordot(np.ones(nc),Rav[:,ip],axes=0)
            dC[:,:,ip] = C_act[:,:,ip] - np.tensordot(np.ones(nc),Cav[:,ip],axes=0)
            dIdR[:,ip] = np.average(dI[:,:,ip]*dR[:,:,ip],weights=pop0,axis=0)
            IR[:,ip] = np.average(I_act[:,:,ip]*R_act[:,:,ip],weights=pop0,axis=0)
            dCdI[:,ip] = np.average(dI[:,:,ip]*dC[:,:,ip],weights=pop0,axis=0)
            CI[:,ip] = np.average(I_act[:,:,ip]*C_act[:,:,ip],weights=pop0,axis=0)
            dI2 [:,ip] = np.average(dI[:,:,ip]*dI[:,:,ip],weights=pop0,axis=0)
            dR2 [:,ip] = np.average(dR[:,:,ip]*dR[:,:,ip],weights=pop0,axis=0)
            dC2 [:,ip] = np.average(dC[:,:,ip]*dC[:,:,ip],weights=pop0,axis=0)
            Cmax[:,ip] = np.max(C_act[:,:,ip],axis=0)

        print ('Filtered maps created.')
        with open('I_act_filtered.pkl', 'wb') as file:
            pickle.dump(I_act, file)
        with open('R_act_filtered.pkl', 'wb') as file:
            pickle.dump(R_act, file)
        with open('C_act_filtered.pkl', 'wb') as file:
            pickle.dump(C_act, file)

        # Fit spectra 
        alpha1=np.zeros(nt_rki)
        alpha0=np.zeros(nt_rki)
        beta1=np.zeros(nt_rki)
        beta0=np.zeros(nt_rki)
        pp0=np.ones((nk,2))
        pp0[:,0]=np.log(p[:])
        
        jpmax = np.where(p>1e7)[0][0]
        
        for i in range(nt_rki):
            alpha0[i]=sm.OLS(np.log(dC2[i,0:jpmax]/Cav[i,0:jpmax]),exog=pp0[0:jpmax,:]).fit().params[0]
            alpha1[i]=sm.OLS(np.log(dC2[i,0:jpmax]/Cav[i,0:jpmax]),exog=pp0[0:jpmax,:]).fit().params[1]
            beta0[i]=sm.OLS(np.log(Cmax[i,:jpmax]-Cav[i,0:jpmax]),exog=pp0[0:jpmax,:]).fit().params[0]
            beta1[i]=sm.OLS(np.log(Cmax[i,:jpmax]-Cav[i,0:jpmax]),exog=pp0[0:jpmax,:]).fit().params[1]
            if (np.any(dIdR[i,:]<=0)):
                alpha1[i]=np.nan
            
        i_nan = np.where(np.isnan(alpha1))
        if len(i_nan[0]>0):
            i_s = 1+np.where((i_nan[0][1:]!=i_nan[0][0:-1]+1))[0]
            i_s = np.insert(i_s,0,0)
            i_e = np.append(i_s[1:]-1,-1)                                                                                                                
            for i_gap in range(len(i_s)):
              ii0 = i_nan[0][i_s[i_gap]]-1
              ii1 = i_nan[0][i_e[i_gap]]+1
              print("Interpolating between data point %i and %i." % (ii0, ii1))
              for ii in range(ii0+1,ii1):
                if ii0<0: ii0 = ii1
                if ii1>=nt_rki: ii1 = ii0
                xi = (ii-ii0)/(ii1-ii0+2)
                alpha1 [ii]   = (1.0-xi)*alpha1 [ii0]  + xi*alpha1 [ii1]

        # Store data
        self.nt = nt_rki
        self.nc = nc
        self.lk = lk
        self.lkID = lkID
        self.date = date_rki
        self.cases = ncase_sym
        self.pop0 = pop0
        self.p = p
        self.Cav = Cav
        self.Iav = Iav
        self.Rav = Rav
        self.dC2 = dC2
        self.dI2 = dI2
        self.dR2 = dR2
        self.CI = CI
        self.IR = IR
        self.dCdI = dCdI
        self.dIdR = dIdR
        self.alpha0 = alpha0
        self.alpha1 = alpha1

    def plot_overview(self):
        plt.plot (pd.to_datetime(self.date, origin = 'julian', unit = 'D'), np.sum(self.cases,axis=0),
                  c='k', label='Daily cases')
        plt.xlabel(r'Date')
        plt.ylabel(r'Daily cases')
        plt.minorticks_on()
        locmax = mdates.AutoDateLocator(minticks=7, maxticks=11)
        locmin = mdates.AutoDateLocator(minticks=12, maxticks=24)
        formatter = mdates.ConciseDateFormatter(locmax)
        plt.gca().xaxis.set_major_locator(locmax)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().xaxis.set_minor_locator(locmin)
        
        plt.ylim(0,40000)

        ax2 = plt.gca().twinx()
        tmp = np.cumsum (np.sum (self.cases, axis=0)) / np.sum(self.pop0[:,0])
        ax2.plot (pd.to_datetime (self.date, unit='D', origin='julian'), tmp, c='r')
        ax2.plot (pd.to_datetime (self.date, unit='D', origin='julian'), 3*tmp, c='r', linestyle = 'dashed')
        ax2.set_ylim(0,0.2)
        ax2.set_ylabel('Cumulative incidence')
        ax2.minorticks_on()
        
        plt.xlim(pd.to_datetime('1/1/2020'),pd.to_datetime('8/1/2021'))
        
        def mark_date (date_str):
            plt.axvline(pd.to_datetime(date_str), c='k', linestyle='dotted')
            plt.text(pd.to_datetime(date_str) + pd.DateOffset(days=0), 0.7, date_str, fontsize=10, \
                     rotation=90, ha='right', va='bottom', transform=plt.gca().get_xaxis_transform())
            
        mark_date('3 Jul 2020')
        mark_date('7 Nov 2020')
        mark_date('27 Nov 2020')
        mark_date('18 Dec 2020')
        mark_date('25 Feb 2021')
        
    
def download (source_url, filename = None):
    if filename == None:
        local_file = 'data/' + source_url.split('/')[-1]
    else:
        local_file = 'data/' + filename
    with requests.get(source_url, stream=True) as response:
        response.raise_for_status()
        with open(local_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=2**16):
                file.write(chunk)
