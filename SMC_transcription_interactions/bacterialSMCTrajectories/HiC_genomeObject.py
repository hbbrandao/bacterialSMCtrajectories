# Created by Hugo Brandao (hbrandao@g.harvard.edu) October 2016
import numpy as np
from mirnylib import plotting
from mirnylib.numutils import ultracorrect
from Bio import SeqIO
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm
from scipy.ndimage.filters import gaussian_filter
import joblib

#cachedir = '.'
#if not os.path.exists(cachedir):
#    os.mkdir(cachedir)
#mem = joblib.Memory(cachedir=cachedir) # cache
mem = joblib.Memory('.') # cache
"""
Functionality: 
HiC_genomDict class will organize HiC_genomeObjects based on a naming classifier: 

HiC_genomeObject will
1) Point to genome annotations fileloc
2) Point to Hi-C maps for specified given genome
3) Contain meta-data such as location of parS sites
4) Point to ChIP tracks and other types of data for given genome
"""

### CACHED FUNCTIONS ###
@mem.cache
def _getOperonList(operon_file):
        operonsBCyc = pd.read_csv(operon_file)
        operonsList = pd.DataFrame(columns=["starts","ends","length","strand"]);

        oend = pd.Series([0]*len(operonsBCyc))
        ostart = pd.Series([0]*len(operonsBCyc))
        ostrand = pd.Series([0]*len(operonsBCyc))
        olen = pd.Series([0]*len(operonsBCyc))
        oexp = pd.Series([0]*len(operonsBCyc))
        for f in range(len(operonsBCyc)):
            gen_coords = [];
            gen_coords += ([int(n) for n in (operonsBCyc["Left-End-Position"].iloc[f]).split('//')])
            gen_coords += ([int(n) for n in (operonsBCyc["Right-End-Position"].iloc[f]).split('//')])
            st = min(gen_coords);
            ed = max(gen_coords); 
            ostart[f] = st
            oend[f] = ed
            olen[f] = np.abs(ed-st)

            if operonsBCyc["Transcription-Direction"].iloc[f] == '+':
                ostrand[f] = 1;
            elif operonsBCyc["Transcription-Direction"].iloc[f] == '-':
                ostrand[f] = -1;    

        operonsList["starts"] = ostart;
        operonsList["ends"] = oend;
        operonsList["length"] = olen;
        operonsList["strand"] = ostrand;
        return operonsList
_getOperonList = mem.cache(_getOperonList)

######################################################################

class HiC_genomeObject: 
    
    def __init__(self):
        self.HiC_genomDict = {}
        self.__mapcount = 0; 
    
    def set_chromosome_length(self,chrmLen): 
        self.HiC_genomDict['chrmLen'] = chrmLen

    def load_text_HiC_Map(self,fname,source='',key='map',iterativeCorrect=False):
        if os.path.isabs(fname):
            assert os.path.exists(fname)    
        elif not os.path.exists(fname):
            assert os.path.exists(os.path.join(source,fname))
            fname = os.path.join(source,fname) 
        self.__mapcount+=1
        if key=='map':
            key='map'+str(self.__mapcount)
            
        try:
            M = np.loadtxt(fname, float)
        
            if iterativeCorrect == True:
                M = ultracorrect(M)/np.median(np.nansum(M,axis=0))
            self.HiC_genomDict[key] = HiC_object();     
            self.HiC_genomDict[key].heatmap  = M;
        except:
            print("could not load file: {0}".format(key))

    def load_csv_ChIP_track(self,expt_file,input_file,source='',key='chip_map'):
        if os.path.isabs(expt_file):
            assert os.path.exists(expt_file)    
        elif not os.path.exists(expt_file):
            assert os.path.exists(os.path.join(source,expt_file))
            expt_file = os.path.join(source,expt_file) 
        
        if os.path.isabs(input_file):
            assert os.path.exists(input_file)    
        elif not os.path.exists(input_file):
            assert os.path.exists(os.path.join(source,input_file))
            input_file = os.path.join(source,input_file)            
            
        self.__mapcount+=1
        if key=='chip_map':
            key='chip_map'+str(self.__mapcount)
        

        self.HiC_genomDict[key] = ChIP_object();     
        
        input_track = pd.read_csv(input_file)
        expt_track = pd.read_csv(expt_file)
        
        self.HiC_genomDict[key].chip_input = input_track
        self.HiC_genomDict[key].chip_raw = expt_track
        self.HiC_genomDict[key].chip_norm  = (expt_track['Value'].values)/(input_track['Value'].values)*np.nansum(input_track['Value'].values)/np.nansum(expt_track['Value'].values);        
        
    def show_heatmap(self,key,vmax=None,vmin=None,showShifted=False,shiftBy=None,logColorScale=None,iterativeCorrect=False,\
                     cmap='fall',fillZeros=True,observedOverExpected=False): 
        # logColorScale is set a floating point value only if we wish to display a log colorbar 
        L = self.HiC_genomDict['chrmLen']//1e3; 
        hmap = self.HiC_genomDict[key].heatmap.copy()
        
        if fillZeros == True:
            hmap[hmap==0] = np.max(hmap)
        
        if observedOverExpected == True: 
            hmap = setMatrilocScaling(hmap,alpha=0)
        
        if iterativeCorrect == True:
            hmap = ultracorrect(hmap) 
            
        if logColorScale != None:
            hmap = np.log2(hmap+logColorScale)
        
        if showShifted==True: 
            if shiftBy is None:
                fshift = L//2; 
                plt.imshow(np.fft.fftshift(np.flipud(hmap)),\
                           extent=[-fshift,fshift,-fshift,fshift],cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
            else:
                hmap = np.flipud(np.roll(np.roll(hmap,-shiftBy,axis=0),-shiftBy,axis=1))
                plt.imshow(hmap, extent=[-shiftBy,L-shiftBy,0-shiftBy,L-shiftBy],\
                                cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')                
        else: 
            plt.imshow(np.flipud(hmap),\
                       extent=[0,L,0,L],cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
    
    def set_parS_sites(self,key,parS_sites=[]):
        assert len(self.HiC_genomDict[key].heatmap)>0
        self.HiC_genomDict[key].parS  = parS_sites;
       
    def load_genBank_annotation(self,gb_file):
        handles = SeqIO.read(open(gb_file), "genbank")  
        colNames = ["starts","ends","strand"]; 
        rRNA_pos = pd.DataFrame(columns=colNames);
        tRNA_pos = pd.DataFrame(columns=colNames);
        gene_pos = pd.DataFrame(columns=colNames);

        for f in range(len(handles.features)):
            df = [];
            feature = handles.features[f]
            if feature.type == 'rRNA':
                st = feature.location.start; 
                ed = feature.location.end; 
                df = pd.DataFrame([[feature.location.start, feature.location.end,
                                    feature.location.strand]], columns=colNames)
                rRNA_pos = rRNA_pos.append(df)
            if feature.type == 'tRNA':
                st = feature.location.start; 
                ed = feature.location.end; 
                df = pd.DataFrame([[feature.location.start, feature.location.end, 
                                    feature.location.strand]], columns=colNames)
                tRNA_pos = tRNA_pos.append(df)
            if feature.type == 'gene':
                st = feature.location.start; 
                ed = feature.location.end; 
                df = pd.DataFrame([[feature.location.start, 
                                    feature.location.end, 
                                    feature.location.strand]], columns=colNames)
                gene_pos = gene_pos.append(df)
        self.HiC_genomDict['rRNA'] = rRNA_pos
        self.HiC_genomDict['tRNA'] = tRNA_pos
        self.HiC_genomDict['gene'] = gene_pos
        
    def load_operonStructure(self,operon_file):
        self.HiC_genomDict["operons"] = _getOperonList(operon_file);
            
    def ang2site(self,ang):
        site = 0; 
        if ang>=0:
            site = ang/360*self.HiC_genomDict['chrmLen'] 
        else:
            site = (360+ang)/360*self.HiC_genomDict['chrmLen'] 
        return int(site)

    def computeDScore(self,mutuallyExclusiveFeatures=True):     
        if 'chrmLen' not in self.HiC_genomDict.keys():
            print('DScore not computed: First upload genome to HiC_genomeObject');
            return;  

        # GET rRNA level transcription
        if 'rRNA' in self.HiC_genomDict.keys():
            dscoreR = np.zeros(self.HiC_genomDict['chrmLen'] );
            for op in range(len(self.HiC_genomDict['rRNA'])):
                st = int(self.HiC_genomDict['rRNA']["starts"].iloc[op])
                ed = int(self.HiC_genomDict['rRNA']["ends"].iloc[op])
                sd = int(self.HiC_genomDict['rRNA']["strand"].iloc[op])
                dscoreR[st:ed] += sd;
            self.HiC_genomDict['dscoreR'] = dscoreR; 
                               
        # GET tRNA level transcription
        if 'tRNA' in self.HiC_genomDict.keys(): 
            dscoreT = np.zeros(self.HiC_genomDict['chrmLen'] );
            for op in range(len(self.HiC_genomDict['tRNA'])):
                st = int(self.HiC_genomDict['tRNA']["starts"].iloc[op])
                ed = int(self.HiC_genomDict['tRNA']["ends"].iloc[op])
                sd = int(self.HiC_genomDict['tRNA']["strand"].iloc[op])
                dscoreT[st:ed] += sd;
                
            # make the features mutually exclusive
            if 'rRNA' in self.HiC_genomDict.keys() and mutuallyExclusiveFeatures==True:
                dscoreT[dscoreR!=0] = 0
                
            self.HiC_genomDict['dscoreT'] = dscoreT; 
            
        # GET operon-level transcription
        if 'operons' in self.HiC_genomDict.keys():
            dscoreO = np.zeros(self.HiC_genomDict['chrmLen'] );
            for op in range(len(self.HiC_genomDict['operons'])):
                st = int(self.HiC_genomDict['operons']["starts"].iloc[op])
                ed = int(self.HiC_genomDict['operons']["ends"].iloc[op])
                sd = int(self.HiC_genomDict['operons']["strand"].iloc[op])
                dscoreO[st:ed] += sd;
                
            # make the features mutually exclusive
            if 'rRNA' in self.HiC_genomDict.keys() and mutuallyExclusiveFeatures==True:
                dscoreO[dscoreR!=0] = 0                
                dscoreO[dscoreT!=0] = 0  
                
            self.HiC_genomDict['dscoreO'] = dscoreO;            
   
    def doExtrusionTrace(self,key,gamma=1,rho=1,tau=1,substeps=1e4,v_avg_bps=833,timeCutoff=50*60, \
                        showShifted=False,doPeriodic=False,showPlot=True, chip_key='',chip_cap=100000,colour='lightblue',lw=4): 

        x_clockwise = []
        x_cclockwise = []

        # get D-scores 
        dO_noshift = self.HiC_genomDict['dscoreO'].copy()
        dR_noshift = self.HiC_genomDict['dscoreR'].copy()
        dT_noshift = self.HiC_genomDict['dscoreT'].copy()  

        if chip_key != '':
            chip_norm = self.HiC_genomDict[chip_key].chip_norm.copy()
            chip_norm[np.isnan(chip_norm)] = 1
            chip_norm[np.isinf(chip_norm)] = 1
            chip_norm[chip_norm==0] = 0
            chip_norm[chip_norm>chip_cap] = chip_cap              
            dO_noshift = np.multiply(dO_noshift, chip_norm )
            dR_noshift = np.multiply(dR_noshift, chip_norm )
            dT_noshift =  np.multiply(dT_noshift, chip_norm )

        # get parS site and chromosome arm lengths
        parS_list = self.HiC_genomDict[key].parS
        L = self.HiC_genomDict['chrmLen'] 
        L_kb  = L//1e3
        ter_site = L//2 #  approximate
        L_arm1 = (L-ter_site)
        L_arm2 = ter_site

        # compute average and maximum speed calibration using the "ori-proximal" experiment as a reference
        # first, compute reference waiting time per feature type per genome locus
        tcO = np.amax(np.c_[-gamma*dO_noshift,dO_noshift],axis=1)
        tccO = np.amax(np.c_[gamma*dO_noshift,-dO_noshift],axis=1)
        tcR = np.amax(np.c_[-rho*dR_noshift,dR_noshift],axis=1)
        tccR = np.amax(np.c_[rho*dR_noshift,-dR_noshift],axis=1)
        tcT = np.amax(np.c_[-tau*dT_noshift,dT_noshift],axis=1)
        tccT = np.amax(np.c_[tau*dT_noshift,-dT_noshift],axis=1)    
        # total waiting time per genome locus
        tclockwise_noshift = tcO + tcR + tcT
        tcounterclockwise_noshift = tccO + tccR + tccT
        # set unannotated regions to the "forward" speed
        tclockwise_noshift[tclockwise_noshift==0] = 1
        tcounterclockwise_noshift[tcounterclockwise_noshift==0] = 1       
        # calibration of relative time to real time
        xi1 = L_arm1/v_avg_bps/ np.sum(tclockwise_noshift[0:L_arm1:1])
        xi2 = L_arm2/v_avg_bps/ np.sum(tcounterclockwise_noshift[L_arm1:-1:1])
        xi = np.sqrt(xi1*xi2)
        # maximum speed of extruder
        vmax = xi*np.sqrt(L_arm1*L_arm2)      

        for parS_idx,parS in enumerate(parS_list):       

            # center D-scores on parS site
            dO = np.roll(dO_noshift.copy(),-int(parS)) 
            dR = np.roll(dR_noshift.copy(),-int(parS))
            dT = np.roll(dT_noshift.copy(),-int(parS))

            # compute waiting time per feature type per genome locus
            tcO = np.amax(np.c_[-gamma*dO,dO],axis=1)
            tccO = np.amax(np.c_[gamma*dO,-dO],axis=1)
            tcR = np.amax(np.c_[-rho*dR,dR],axis=1)
            tccR = np.amax(np.c_[rho*dR,-dR],axis=1)
            tcT = np.amax(np.c_[-tau*dT,dT],axis=1)
            tccT = np.amax(np.c_[tau*dT,-dT],axis=1)    

            # compute waiting time per genome locus
            tclockwise = tcO.copy() + tcR.copy() + tcT.copy()
            tcounterclockwise = tccO.copy() + tccR.copy() + tccT.copy()

            # set unannotated regions to "forward" speed
            tclockwise[tclockwise==0] = 1
            tcounterclockwise[tcounterclockwise==0] = 1   

            # compute cumulative sum (in the correct orientations)
            T_clockwise = xi*np.cumsum(tclockwise)
            T_counterclockwise = xi*np.cumsum(tcounterclockwise[::-1])

            maxT = max([T_clockwise[-1],T_counterclockwise[-1]]) # maximum time to traverse genome
            maxT = min([maxT,timeCutoff])
            tq = np.arange(0,maxT,maxT/substeps) # query times (break into # substeps)

            x_clockwise.append(np.interp(tq,T_clockwise,np.arange(len(T_clockwise))/1e3)) # in kb
            x_cclockwise.append(np.interp(tq,T_counterclockwise,np.arange(len(T_counterclockwise))/1e3)) # in kb   


            if showShifted == True:
                fshift = L_kb//2;
            else:
                fshift = 0; 

            if doPeriodic==True:    
                X_coords = (-x_cclockwise[parS_idx]+parS//1e3-fshift)%L_kb-fshift
                Y_coords = (x_clockwise[parS_idx]+parS//1e3-fshift)%L_kb-fshift
            else: 
                if np.sign(L//2 - parS) <0: 
                    X_coords = (-x_cclockwise[parS_idx]+parS//1e3-fshift)-fshift
                    Y_coords = (x_clockwise[parS_idx]+parS//1e3-fshift)-fshift
                else: 
                    X_coords = (-x_cclockwise[parS_idx]+parS//1e3-fshift)-fshift
                    Y_coords = (x_clockwise[parS_idx]+parS//1e3-fshift)-fshift
            if showPlot==True:
                plt.plot(X_coords,Y_coords,'-',color=colour,linewidth=lw); 
                plt.draw()
        return x_clockwise, x_cclockwise, tq, X_coords, Y_coords, vmax    
    
    def doExtrusionTraceByStep(self,key,fmiloc=0,gamma=1,rho=1,tau=1,substeps=1e4,v_avg_bps=833,timeCutoff=50*60, \
                        showShifted=False,doPeriodic=False,showPlot=True, chip_key='',chip_cap=100000,colour='lightblue',lw=4): 

        x_clockwise = []
        x_cclockwise = []

        # get D-scores 
        dO_noshift = self.HiC_genomDict['dscoreO'].copy()
        dR_noshift = self.HiC_genomDict['dscoreR'].copy()
        dT_noshift = self.HiC_genomDict['dscoreT'].copy()    

        if chip_key != '':
            chip_norm = self.HiC_genomDict[chip_key].chip_norm.copy()
            chip_norm[np.isnan(chip_norm)] = 1
            chip_norm[np.isinf(chip_norm)] = 1
            chip_norm[chip_norm==0] = 0
            chip_norm[chip_norm>chip_cap] = chip_cap              
            dO_noshift = np.multiply(dO_noshift, chip_norm )
            dR_noshift = np.multiply(dR_noshift, chip_norm )
            dT_noshift =  np.multiply(dT_noshift, chip_norm )

        # get parS site and chromosome arm lengths
        parS_list = self.HiC_genomDict[key].parS
        L = self.HiC_genomDict['chrmLen'] 
        L_kb  = L//1e3
        ter_site = L//2 #  approximate
        L_arm1 = (L-ter_site)
        L_arm2 = ter_site

        # compute average and maximum speed calibration using the "ori-proximal" experiment as a reference
        # first, compute reference waiting time per feature type per genome locus
        tcO = np.amax(np.c_[-gamma*dO_noshift,dO_noshift],axis=1)
        tccO = np.amax(np.c_[gamma*dO_noshift,-dO_noshift],axis=1)
        tcR = np.amax(np.c_[-rho*dR_noshift,dR_noshift],axis=1)
        tccR = np.amax(np.c_[rho*dR_noshift,-dR_noshift],axis=1)
        tcT = np.amax(np.c_[-tau*dT_noshift,dT_noshift],axis=1)
        tccT = np.amax(np.c_[tau*dT_noshift,-dT_noshift],axis=1)    
        # total waiting time per genome locus
        tclockwise_noshift = tcO + tcR + tcT
        tcounterclockwise_noshift = tccO + tccR + tccT
        # set unannotated regions to the "forward" speed
        tclockwise_noshift[tclockwise_noshift==0] = 1
        tcounterclockwise_noshift[tcounterclockwise_noshift==0] = 1       
        # calibration of relative time to real time
        xi1 = L_arm1/v_avg_bps/ np.sum(tclockwise_noshift[0:L_arm1:1])
        xi2 = L_arm2/v_avg_bps/ np.sum(tcounterclockwise_noshift[L_arm1:-1:1])
        xi = np.sqrt(xi1*xi2)
        # maximum speed of extruder
        vmax = xi*np.sqrt(L_arm1*L_arm2)      

        for parS_idx,parS in enumerate(parS_list):       

            # center D-scores on parS site
            dO = np.roll(dO_noshift.copy(),-int(parS)) 
            dR = np.roll(dR_noshift.copy(),-int(parS))
            dT = np.roll(dT_noshift.copy(),-int(parS))

            # compute waiting time per feature type per genome locus
            tcO = np.amax(np.c_[-gamma*dO,dO],axis=1)
            tccO = np.amax(np.c_[gamma*dO,-dO],axis=1)
            tcR = np.amax(np.c_[-rho*dR,dR],axis=1)
            tccR = np.amax(np.c_[rho*dR,-dR],axis=1)
            tcT = np.amax(np.c_[-tau*dT,dT],axis=1)
            tccT = np.amax(np.c_[tau*dT,-dT],axis=1)    

            # compute waiting time per genome locus
            tclockwise = tcO.copy() + tcR.copy() + tcT.copy()
            tcounterclockwise = tccO.copy() + tccR.copy() + tccT.copy()

            # set unannotated regions to "forward" speed
            tclockwise[tclockwise==0] = 1
            tcounterclockwise[tcounterclockwise==0] = 1   

            # set the correct orientation for counter-clockwise steps and convert to real times
            tclockwise = xi*tclockwise
            tcounterclockwise = xi*tcounterclockwise[::-1]

            # compute cumulative sum, step by step (clockwise and counter-clockwise)
            tc = 0 # cumulative time for clockwise steps
            tcc = 0 # cumulative time for counter-clockwise steps
            xc = 0
            xcc = 0
            xyt_pairs = []
            while (xc < L) and (xcc < L) and ((tc<timeCutoff)  or (tcc<timeCutoff)):

                # step xc
                if tc <= tcc: 
                    xc = xc + 1
                    tc += fmiloc/2*tcounterclockwise[int(xcc)] + (1-fmiloc/2)*tclockwise[int(xc)]

                if tcc < tc:
                    xcc = xcc+ 1
                    tcc += fmiloc/2*tclockwise[int(xc)] + (1-fmiloc/2)*tcounterclockwise[int(xcc)]

                #print((np.min([tc,tcc]),xc,xcc))
                xyt_pairs.append([np.min([tc,tcc]),xc,xcc])

            xyt_pairs = np.asarray(xyt_pairs)

            maxT = xyt_pairs[-1,0] # maximum time to traverse genome
            maxT = min([maxT,timeCutoff])
            tq = np.arange(0,maxT,maxT/substeps) # query times (break into # substeps)

            x_clockwise.append(np.interp(tq,xyt_pairs[:,0],(xyt_pairs[:,1])/1e3)) # in kb
            x_cclockwise.append(np.interp(tq,xyt_pairs[:,0],(xyt_pairs[:,2])/1e3)) # in kb   


            if showShifted == True:
                fshift = L_kb//2;
            else:
                fshift = 0; 

            if doPeriodic==True:    
                X_coords = (-x_cclockwise[parS_idx]+parS//1e3-fshift)%L_kb-fshift
                Y_coords = (x_clockwise[parS_idx]+parS//1e3-fshift)%L_kb-fshift
            else: 
                if np.sign(L//2 - parS) <0: 
                    X_coords = (-x_cclockwise[parS_idx]+parS//1e3-fshift)-fshift
                    Y_coords = (x_clockwise[parS_idx]+parS//1e3-fshift)-fshift
                else: 
                    X_coords = (-x_cclockwise[parS_idx]+parS//1e3-fshift)-fshift
                    Y_coords = (x_clockwise[parS_idx]+parS//1e3-fshift)-fshift
            if showPlot==True:
                plt.plot(X_coords,Y_coords,'-',color=colour,linewidth=lw); 
                plt.draw()
        return x_clockwise, x_cclockwise, tq, X_coords, Y_coords, vmax    


    def show_chip(self,chip_key,chip_cap=100000,gauss_filt_length=1000,vmax=None,vmin=None,showShifted=False,shiftBy=None): 

        L = self.HiC_genomDict['chrmLen'] //1e3; 

        chip_norm = self.HiC_genomDict[chip_key].chip_norm.copy()
        chip_norm[np.isnan(chip_norm)] = 1
        chip_norm[np.isinf(chip_norm)] = 1
        chip_norm[chip_norm==0] = 0
        chip_norm[chip_norm>chip_cap] = chip_cap     


        if showShifted==True: 
            if shiftBy is None:
                fshift = L//2; 
                plt.plot(np.fft.fftshift(gaussian_filter(chip_norm,gauss_filt_length)),\
                         label='{} bp window average'.format(gauss_filt_length))
                plt.xlim([-len(chip_norm)//2,len(chip_norm)//2])  

            else:
                plt.plot(np.roll(gaussian_filter(chip_norm,gauss_filt_length),-int(shiftBy)),\
                         label='{} bp window average'.format(gauss_filt_length))
                plt.xlim([-int(shiftBy),len(chip_norm)-int(shiftBy)]) 

        else: 
            plt.plot(np.roll(gaussian_filter(chip_norm,gauss_filt_length),0),\
                     label='{} bp window average'.format(gauss_filt_length))
            plt.xlim([0,len(chip_norm)]) 

        plt.xlabel('Genome position')
        plt.ylabel('chrmLenChIP signal (arb.)')
    
# this is a container that holds heatmaps and associated metadata
class HiC_object: 
    def __init__(self):
        self.heatmap =[];
        self.parS = []  
        self.genotypeInfo = []

class ChIP_object: 
    def __init__(self):
        self.ChIP_track = []
        self.expt_track = []
        self.input_track = []
        self.genotypeInfo = []      

# these are helper functions that return standard utility outputs
from mirnylib.numutils import logbinsnew, fillDiagonal
def getLogBinnedScaling(inMatriloc,measureType='sum',isCircular=True):
    inMatriloc = np.array(inMatriloc, dtype = np.double)
    N = len(inMatriloc)    
        
    marginals = np.sum(inMatriloc, axis=0)
    mask = marginals > 0 
    mask2d = mask[:,None] * mask[None,:]
    
    
    bins = logbinsnew(1, N, 1.3)
    mids = 0.5 * (bins[:-1] + bins[1:])
    Pc = [] 
    for st, end in zip(bins[:-1], bins[1:]):
        curmean = 0
        maskmean = 0
        for i in range(st, end):
            if measureType == 'median':
                curmean += np.median(np.diagonal(inMatriloc, i))
                maskmean += np.median(np.diagonal(mask2d, i))
                if (isCircular==True) and (i>0): # to not double-count the main diagonal if considered
                    curmean += np.median(np.diagonal(inMatriloc, N-i))
                    maskmean += np.median(np.diagonal(mask2d, N-i))
            else:
                curmean += np.nanmean(np.diagonal(inMatriloc, i))
                maskmean += np.nanmean(np.diagonal(mask2d, i))
                if (isCircular==True) and (i>0): 
                    curmean += np.median(np.diagonal(inMatriloc, N-i-1))
                    maskmean += np.median(np.diagonal(mask2d, N-i-1))
        Pc.append(curmean / maskmean)    
    mids = np.r_[mids, N]
    Pc = np.r_[Pc, np.sqrt((Pc[-1] / Pc[-2])) * Pc[-1]]
    if isCircular == True:
        Pc_interp = np.interp(np.arange(N),mids,Pc)
        Pc_new = np.zeros(N)
        mids = np.arange(N) # re-write mids
        for i in range(0,N):
            if i> (N-i):
                break
            Pc_new[i] = Pc_interp[i]
            if i>0:
                Pc_new[N-i-1] = Pc_interp[i]
            
    else:
        Pc_new = Pc
    return Pc_new, mids

def setMatrilocScaling(inMatriloc, alpha=0):
    inMatriloc = inMatriloc.copy()
    N = len(inMatriloc)
    Pc, mids = getLogBinnedScaling(inMatriloc,isCircular=True)
    for i in range(N):
        fillDiagonal(inMatriloc,np.diagonal(inMatriloc,i)/Pc[i]/(i**(-alpha)),i)

    return np.triu(inMatriloc)+np.triu(inMatriloc).T

