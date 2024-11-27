import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Arc
import pandas as pd

import os 
import shutil

import contact_map_generator_from_shortest_path as cmgsp
from contact_map_generator_from_shortest_path import zoomArray

import sys
sys.setrecursionlimit(4000)

from multiprocessing import Pool
import pickle

from matplotlib.gridspec import GridSpec
from itertools import product
import matplotlib.colors as clr 
import matplotlib

from brandaolib.plotting import tung_map
class smcTranslocator():
    
    def __init__(self, 
                 birth_prob, 
                 basal_death_prob, 
                 stall_prob_left, 
                 stall_prob_right, 
                 step_prob_left, 
                 step_prob_right, 
                 stall_death_prob, 
                 prob_falloff_ori2ter_ter2ori_bypass,
                 N, # size of chromosome
                 replication_fork_loci, # left/right positions of the replication fork
                 numSmc,
                 smc_pairs=None,
                 leading_strand_preference=0.5,
                ):
     
        self.N = N 
        self.M = numSmc
        self.birth_prob = birth_prob
        
        self.stall_prob_left = stall_prob_left
        self.stall_prob_right = stall_prob_right
        self.falloff = basal_death_prob
        
        self.step_prob_left = step_prob_left
        self.step_prob_right = step_prob_right
        cumem = np.cumsum(birth_prob)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.loading_probability = np.array(cumem, np.double)
        
        self.replication_fork_loci = replication_fork_loci
        
        if smc_pairs == None:
            self.SMCs1 = np.zeros((self.M), int)
            self.SMCs2 = np.zeros((self.M), int)
            self.occupied = np.zeros(2*self.N+1, int)
        else:
            self.SMCs1 = np.array([p[0] for p in smc_pairs], int)
            self.SMCs2 = np.array([p[1] for p in smc_pairs], int)           
            self.occupied = np.zeros(2*self.N+1, int)
            for p in smc_pairs:
                self.occupied[p[0]] += 1
                self.occupied[p[1]] += 1
            
        self.leading_strand_preference = leading_strand_preference
        
        self.stalled1 = np.zeros(self.M, int)
        self.stalled2 = np.zeros(self.M, int)
        
        self.stallFalloff = stall_death_prob
        self.knockOffProb_OriToTer = prob_falloff_ori2ter_ter2ori_bypass[0] # rate of facilitated dissociation
        self.knockOffProb_TerToOri = prob_falloff_ori2ter_ter2ori_bypass[1] # rate of facilitated dissociation
        self.kBypass = prob_falloff_ori2ter_ter2ori_bypass[2] # rate of bypassing
        
        self.maxss = 1000000
        self.curss = 99999999

        # only pre-initialize if the initial conditions are not specified
        if smc_pairs == None:
            for ind in np.arange(self.M):
                self.birth(ind)

        # create locus classificaiton array
        self.locus_classification = self.classify_loci()
        
        
    def birth(self, ind):
  
        while True:
            pos = self.getss()
            if pos > 2*self.N - 1: 
                # this last array position is part of the "reservoir" for the SMCs
                # or otherwise the equivalent to the pool of cytosolic SMCs
                self.SMCs1[ind] = 2*self.N
                self.SMCs2[ind] = 2*self.N
                self.occupied[2*self.N] += 1 
                self.occupied[2*self.N] += 1     
                return                

            elif pos < 0: 
                print("bad value", pos, self.loading_probability[0])
                continue 
 
            
            # if binding is to the maternal chromosome
            elif 0 <= pos < self.N:
                # ensure we do not have multiple SMCs co-occupying to the same locus
                if (self.occupied[pos] >= 1) or (self.occupied[np.mod(pos+1,self.N)] >= 1): 
                    continue

                self.SMCs1[ind] = pos
                self.SMCs2[ind] = np.mod(pos+1,self.N)
                self.occupied[pos] += 1 
                self.occupied[np.mod(pos+1,self.N)] += 1                
                
            # if binding to the daughter chromosome
            elif self.N <= pos < 2*self.N:
                  
                # get the position "to the right"
                pos2 = np.mod(pos+1,self.N) + self.N
                
                # if either position is in the unreplicated daughter strand, map it to maternal strand
                if self.is_locus_replicated(pos)==False:
                    pos = np.mod(pos, self.N)
                if self.is_locus_replicated(pos2)==False:
                    pos2 = np.mod(pos2, self.N)
                    
                if (self.occupied[pos] >= 1) or (self.occupied[pos2] >= 1): 
                    continue

                self.SMCs1[ind] = pos
                self.SMCs2[ind] = pos2
                self.occupied[pos] += 1 
                self.occupied[pos2] += 1                   
            else:
                assert 1==0
                    
            
            return

    def death(self):

        for i in np.arange(self.M):
            if self.stalled1[i] == 0:
                falloff1 = self.falloff[self.SMCs1[i]]
            else: 
                falloff1 = self.stallFalloff[self.SMCs1[i]]
            if self.stalled2[i] == 0:
                falloff2 = self.falloff[self.SMCs2[i]]
            else:
                falloff2 = self.stallFalloff[self.SMCs2[i]]              
            
            falloff = max(falloff1, falloff2)
            if np.random.rand() < falloff:                 
                self.occupied[self.SMCs1[i]] -= 1  
                self.occupied[self.SMCs2[i]] -= 1  
                self.stalled1[i] = 0
                self.stalled2[i] = 0
                self.birth(i)
    
    def getss(self):
        # precomputes an array of random loading positions assigned via loading_probability
        if self.curss >= self.maxss - 1:
            foundArray = np.array(np.searchsorted(self.loading_probability, 
                                                  np.random.random(self.maxss)), 
                                  dtype = np.long)
            self.ssarray = foundArray
            self.curss = -1
        # look up next random loading position
        self.curss += 1         
        return self.ssarray[self.curss]
 
    def is_locus_replicated(self, pos):
        return (self.replication_fork_loci[0] <= np.mod(pos, self.N) <= self.replication_fork_loci[1])==False
        
    def is_locus_maternal_strand(self,pos):
        return pos < self.N
    
    def is_locus_daughter_strand(self,pos):
        return self.N <= pos <= 2*self.N-1
        
    def classify_loci(self):        
        classifications = np.zeros_like(self.birth_prob)
        # RM: 1: if position is on replicated maternal strand        
        # RD: 2: if position is on replicated daugther strand        
        # UM: 3: if position is on unreplicated maternal strand        
        # UD: 4: if position is on unreplicated daughter strand        
        # CT: 0: if position is cytosolic        
        for pos in range(len(classifications)):
            code = 0
            # mother strand
            if self.is_locus_maternal_strand(pos):
                
                if self.is_locus_replicated(pos):
                    code = 1                
                else:
                    code = 3      
            # daughter strand
            elif self.is_locus_daughter_strand(pos):
                
                if self.is_locus_replicated( pos):
                    code = 2                
                else:
                    code = 4
            
            classifications[pos] = code
        return classifications
    
    
    def get_next_position(self, pos, direction):
        
        # get old locus type
        old_locus_type = self.locus_classification[pos]
        
        if old_locus_type == 0: # if cytosolic
            return pos
        
        
        # if the old locus is on the unreplicated daughter strand, map it to mother strand
        if old_locus_type == 4: 
            pos = np.mod(pos,self.N)
            old_locus_type = 3
        
        
        # get new locus type, and tentative new position
        new_mod_pos = np.mod(pos+direction, self.N)
        if old_locus_type == 2 or old_locus_type ==4: # if daugther 
            new_pos = new_mod_pos + self.N
        else: # if mother 
            new_pos = new_mod_pos      
        new_locus_type = self.locus_classification[new_pos]
        
        # if no locus type change, keep new position
        if old_locus_type == new_locus_type:
            return new_pos
        
        
        # if unreplicated daugther strand - not an allowed position - map to unreplicated mother strand
        if new_locus_type == 4: 
            return new_mod_pos
        
        # if unreplicated mother strand - no futher decisions needed
        if new_locus_type == 3: 
            return new_pos
        
        # if on replicated daughter strand
        if new_locus_type == 2:
            raise AttributeError
            # this should never happen!
        
        if new_locus_type == 1: 
            assert old_locus_type == 3
            # flip a coin as to whether the movement goes to the mother or daughter strand
            
            # find whether SMC is closer to the right or left replication fork
            closest_fork = np.argmin(np.abs(new_pos - np.array(self.replication_fork_loci)))
            
            # if the new position is the first fork position
            if closest_fork == 0:
                # leading strand is the mother chromosome
                if np.random.rand() <= self.leading_strand_preference:
                    return new_pos
                else:
                    return new_pos + self.N                
            else:
                # leading strand is the daughter chromosme
                if np.random.rand() <= self.leading_strand_preference:
                    return new_pos + self.N 
                else:
                    return new_pos                 
        
        
    def step(self):

        for i in range(self.M):   
            stall1 = self.stall_prob_left[self.SMCs1[i]]
            stall2 = self.stall_prob_right[self.SMCs2[i]]
                                                      
            # update positions             
            cur1 = self.SMCs1[i]
            cur2 = self.SMCs2[i]
            
            # if SMC position is in the pool of cytosolic SMCs, skip it
            if (cur1 >= 2*self.N) or (cur2 >= 2*self.N):
                continue
            
            # new positions with the periodic boundaries
            newCur1 = self.get_next_position(cur1,-1) # left 
            newCur2 = self.get_next_position(cur2, 1) # right 
            
            assert newCur1 <= 2*self.N-1
            assert newCur2 <= 2*self.N-1
            
            # reset "is stalled" -> this actually means "will dissociate from collision"
            if stall1 != 0:
                self.stalled1[i] = 1
            if stall2 != 0:
                self.stalled2[i] = 1               
            
            # move each side only if the other side is "free" or stall/knock off
            if self.stalled1[i] == 0 and stall1 == 0: 
                # bypass
                step_prob = self.step_prob_left[self.SMCs1[i]]
                if np.random.rand() <= step_prob: 
                    if (self.occupied[newCur1] == 0):
                        # take a normal step 
                        self.occupied[newCur1] += 1
                        self.occupied[cur1] -= 1
                        self.SMCs1[i] = newCur1  
                    else:
                        rateSum = self.knockOffProb_OriToTer+self.kBypass
                        if np.random.rand() <= rateSum: 
                            if np.random.rand() <= self.knockOffProb_OriToTer/rateSum:
                                # stall and maybe dissociate
                                self.stalled1[i] = 1
                            else:
                                self.occupied[newCur1] += 1
                                self.occupied[cur1] -= 1
                                self.SMCs1[i] = newCur1                              
                        
            if self.stalled2[i] == 0 and stall2 == 0:  
                step_prob = self.step_prob_right[self.SMCs2[i]]
                if np.random.rand() <= step_prob:                 
                    if (self.occupied[newCur2] == 0) :        
                        # take a normal step 
                        self.occupied[newCur2] += 1
                        self.occupied[cur2] -= 1
                        self.SMCs2[i] = newCur2
                    else:
                        rateSum = self.knockOffProb_TerToOri+self.kBypass
                        if np.random.rand() <= rateSum:                     
                            if np.random.rand() <= self.knockOffProb_TerToOri/rateSum:
                                # stall and maybe dissociate
                                self.stalled2[i] = 1     
                            else:
                                # bypass
                                self.occupied[newCur2] += 1
                                self.occupied[cur2] -= 1
                                self.SMCs2[i] = newCur2                            
                    
            # mark for dissociation if either side is stalled      
            if  (self.stalled2[i] == 1) or (self.stalled1[i] == 1):
                self.stalled1[i] = 1
                self.stalled2[i] = 1             
        
    def steps(self,N):
        for i in np.arange(N):
            self.death()
            self.step()
            
    def getOccupied(self):
        return np.array(self.occupied)
    
    def getSMCs(self):
        return np.array(self.SMCs1), np.array(self.SMCs2)
        
        
    def updateMap(self, cmap):
        cmap[self.SMCs1, self.SMCs2] += 1
        cmap[self.SMCs2, self.SMCs1] += 1

    def updatePos(self, pos, ind):
        pos[ind, self.SMCs1] = 1
        pos[ind, self.SMCs2] = 1



def plot_arcs(SMCTran,color=0,cmap=plt.cm.gnuplot2_r):
    N = SMCTran.N
    plt.xlim([0,2*N/10])
    plt.ylim([0,2000/10])


    p1, p2 = SMCTran.getSMCs()
    for r, l in zip(p1,p2):

        if r == l:
            continue
        center_y = 0
        center_x = (r+l)/2/10
        width = np.abs(r-l)/10

        patch = Arc((center_x,center_y),width,width/2,theta2=180,lw=2,color=cmap(color))

        ax.add_patch(patch)


    plt.axvline(SMCTran.replication_fork_loci[0]/10)
    plt.axvline(SMCTran.replication_fork_loci[1]/10)
    plt.axvline((SMCTran.replication_fork_loci[0]+N)/10 )
    plt.axvline((SMCTran.replication_fork_loci[1]+N)/10 )
    plt.axvline(N/10,color='r')

    
def get_replisome_position(time_samples=[15], # time in mins to stop the DNAP translocation
                           L=int(4.04e6), # length in bp
                           dx=1000, # discretization in bp
                           DNAP_load_time=10, # DNAP load time in min
                           initiation_time=15, # experiment initiation time in min (time allowed for DNAP loading)
                           prob_stall=0.0004, # probability per step of stalling
                           shift_time=0, # time when the DNAP speed changes (e.g. due to temp shift)
                           fast_rate=3,
                           slow_rate=0.85,
                          ): 

    prob_DNAP_loads = 1-np.exp(-initiation_time/DNAP_load_time) # probability of loading DNAP

    # probability that DNAP steps forward
    step_time_dict = {True: np.ones(L//dx)*slow_rate, False: np.ones(L//dx)*fast_rate} # true for post-shift
    
    # array of replisome positions for the list of sample times
    replisome_position = []    
    for total_time in [t*60 for t in time_samples]:  
        loading_time = np.random.exponential(DNAP_load_time)
        is_loaded = (loading_time <= initiation_time)
        
        if is_loaded == False:
            replisome_position.append(np.nan)
            continue

        
        this_time = -initiation_time*60
        for pos in range(L//dx):

            step_time = step_time_dict[this_time>=shift_time]
            step_time_val = step_time[pos] 
            this_time += np.random.exponential(step_time_val) # < step_probability[pos]:
   

            if this_time + loading_time*60 > total_time:
                break

            # stall DNAP
            if np.random.rand()<= prob_stall:
                #pos >= this_time >= DNAP_stall_time*60:
                break

            # only allow DNAP to replicate 1/2 of the chromosome
            if pos > L//dx//2:
                pos=L//dx//2
                break

        replisome_position.append(pos)        

    return replisome_position

def get_replisome_position_distribution(
            time_sample=25, # time in mins to query DNA translocation from the start of the experiment
            L=4040, # genome length in kb
            DNAP_load_time=10, # DNAP loading time constant in min
            initiation_time=15, # time allowed for DNAP loading 
            shift_time=15, # time when the DNAP speed changes (e.g. due to temp shift)
            DNAP_stall_time = 41, # time-constant in min for DNAP spontaneously stalling translocation 
            low_temp_kb_per_min=40,
            high_temp_kb_per_min=60,
            num_samples=100,
            dx=10, # position downsampling factor
            fraction_pre_stalled=0.2,
    ): 


    # array of replisome positions for the list of sample times
    replisome_position = []    
    
    # draw from distribution of loading times
    u = np.linspace(0,0.999,num_samples)

    loading_time_samples = -DNAP_load_time*np.log(1-u)
    stalling_time_samples = -DNAP_stall_time*np.log(1-u)

    total_time = time_sample
    
    times_list = []
    for loading_time in loading_time_samples:
        for stalling_time in stalling_time_samples:
            is_loaded = (loading_time <= np.min([initiation_time, time_sample]))
            if is_loaded == False:
                replisome_position.append([np.nan,np.nan])
                continue


            max_time = np.min([stalling_time, total_time-loading_time])


            translocation_time_before_shift = shift_time - loading_time
            translocation_time_post_shift = max_time - translocation_time_before_shift
            assert translocation_time_before_shift >= 0

            # nominal distance travelled by DNAP    
            if translocation_time_post_shift < 0:
                translocation_time_post_shift = 0
                translocation_time_before_shift = max_time

            distance = (low_temp_kb_per_min*translocation_time_before_shift + 
                       high_temp_kb_per_min*translocation_time_post_shift)

            pos = distance//dx
            if pos > L/2/dx:
                pos=(L/dx)//2
            replisome_position.append([int(pos), int(np.mod(L/dx-pos,L/dx))])
            
    # add some fraction of pre-stalled replisomes
    if fraction_pre_stalled>0:
        num_pre_stalled = int(len(replisome_position)*fraction_pre_stalled/(1-fraction_pre_stalled))
        u = np.linspace(0,1,num_pre_stalled)
        pre_stall_locations = 0.5*(L/dx)*u**2
        for pos in pre_stall_locations:
            replisome_position.append([int(pos), int(np.mod(L/dx-pos,L/dx))])
    
    return replisome_position

def init_model(
    smc_pairs = None,
    N = 4040,
    replication_fork_position = [1000, 3500],
    cytosolic_buffer_length = 1,
    cytosolic_lifetime = 5*60, # seconds
    cytosolicStrength = 4*4040,
    parS_sites = [-1, 30],
    parS_strengths = [2*4040, 2*4040],
    BASE_STOCHASTICITY = 0.05,
    LIFETIME = 2000,
    ter_to_ori_slowdown = 0.65, 
    prob_falloff_ori2ter_ter2ori_bypass = [0.01,0.01,0.04],
    smcNum_bound = 30, 
    smcNum_free = 10,   
    fork_pause_location = None,
    fork_pause_time = 0,
    ter_unloading_time = 0.6,
    leading_strand_preference = 0.5,
    stall_at_replication_forks = False,
    fork_unloading_time = 1,
    ):

    # set unloading rate in region near terminus
    unloading_sites = [x for x in np.arange(1950,2050)]
    unloading_probability = [0.1*BASE_STOCHASTICITY ]*len(unloading_sites) 
    
    
    # get number of SMCs
    if smc_pairs is None:
        smcNum = smcNum_bound+ smcNum_free
    else:
        smcNum = len(smc_pairs)

    # set the loading probability
    birth_array = np.ones(2*N + cytosolic_buffer_length) # sets the loading rate at each chromosomal position


    # set the proability of stalling and stepping
    stall_prob_left = np.zeros(2*N + cytosolic_buffer_length) # probability of stalling moving left
    stall_prob_right = np.zeros(2*N + cytosolic_buffer_length) # probability of stalling moving right
    step_prob_left = np.ones(2*N + cytosolic_buffer_length)*BASE_STOCHASTICITY # probability of stepping left
    step_prob_right = np.ones(2*N + cytosolic_buffer_length)*BASE_STOCHASTICITY # probability of stepping right


    # set baseline translocation asymmetry
    diff_site = 1905586//1000
    step_prob_left[90:diff_site] *= ter_to_ori_slowdown
    step_prob_right[diff_site:N] *= ter_to_ori_slowdown
    step_prob_right[0:90] *= ter_to_ori_slowdown    
    step_prob_left[N+90:N+90+diff_site] *= ter_to_ori_slowdown
    step_prob_right[N+diff_site:2*N] *= ter_to_ori_slowdown
    step_prob_right[N:N+90] *= ter_to_ori_slowdown        

    # set the basal unloading rate 
    death_array = np.ones(2*N + cytosolic_buffer_length)/LIFETIME*BASE_STOCHASTICITY

    # set terminus unloading probability
    for i, s in zip(unloading_sites, unloading_probability):
        death_array[i] = s

    # set the probability that a stalled SMC will dissociate    
    stall_death_array = np.ones(2*N + cytosolic_buffer_length) 

    # set cytosolic pool binding strength 
    # i.e., probability of "binding" (going to) the cytoplasm instead of the chromosome
    birth_array[2*N:2*N+cytosolic_buffer_length] = cytosolicStrength/cytosolic_buffer_length 
    death_array[2*N:2*N+cytosolic_buffer_length] = BASE_STOCHASTICITY/cytosolic_lifetime 
    step_prob_left[2*N:2*N+cytosolic_buffer_length] = 0 
    step_prob_right[2*N:2*N+cytosolic_buffer_length] = 0

    # set locations of SMC slowing down (or stalling)
    for p in fork_pause_location:
        step_prob_left[p:p+1] = BASE_STOCHASTICITY/fork_pause_time
        step_prob_right[p:p+1] = BASE_STOCHASTICITY/fork_pause_time
        step_prob_left[N+p:N+p+1] = fork_pause_time
        step_prob_right[N+p:N+p+1] = fork_pause_time

        # if unloading probability at the fork 
        death_array[p:p+1] = BASE_STOCHASTICITY/fork_unloading_time
        death_array[N+p:N+p+1] = BASE_STOCHASTICITY/fork_unloading_time
        if stall_at_replication_forks is True:
            stall_prob_left[p:p+1] = 1
            stall_prob_right[p:p+1] = 1
            stall_prob_left[N+p:N+p+1] = 1
            stall_prob_right[N+p:N+p+1] = 1            

    # set default parS strength
    if parS_strengths is None:
        parS_strengths = [4*N/len(parS_sites)]*len(parS_sites)    

    # set parS locations 
    for s, ss in zip(parS_sites,parS_strengths):   
        parS_location = np.mod(int(s/360*N),N) 
        birth_array[parS_location] = ss
        # check if chromosome segment is replicated
        if parS_location < np.min(replication_fork_position) or parS_location > np.max(replication_fork_position):
            birth_array[parS_location + N] = ss

    # zero out the translocation and binding probability at the second terminus region
    birth_array[N+replication_fork_position[0]:N+replication_fork_position[1]] = 0


    # create initial conditions for the SMCs, assume totally random to start, unless pre-set
    if smc_pairs is None:
        # initialize the SMCs
        free_smc_pairs = [(2*N,2*N)]*int(smcNum_free)
        bound_smc_pairs = []
        for c in np.random.randint(0,2,smcNum_bound):
            pos1, pos2 = np.sort([np.random.randint(N) + c*N, np.random.randint(N) + c*N])

            if c > 0:
                if replication_fork_position[0] < np.mod(pos1,N-1) < replication_fork_position[1]:
                    pos1 = np.mod(pos1,N)
                if replication_fork_position[0] < np.mod(pos2,N-1) < replication_fork_position[1]:
                    pos2 = np.mod(pos2,N)   
            bound_smc_pairs.append((pos1,pos2))

        smc_pairs = free_smc_pairs + bound_smc_pairs

    # create object              
    SMCTran = smcTranslocator(birth_array, 
                             death_array, 
                             stall_prob_left, 
                             stall_prob_right, 
                             step_prob_left, 
                             step_prob_right, 
                             stall_death_array, 
                             prob_falloff_ori2ter_ter2ori_bypass,
                             N, # size of chromosome
                             replication_fork_position, # left/right positions of the replication fork
                             smcNum,
                             smc_pairs=smc_pairs, 
                             leading_strand_preference= leading_strand_preference, 
                            )
    return SMCTran


import random
import string
def make_save_file_name(params_dict,verbose=True):
    return ''.join(random.choices(string.ascii_uppercase, k=3)) + ''.join(random.choices(string.digits, k=3))
    
    
    
def gridspec_inches(
    wcols,
    hrows,
    wspace=0.75,
    hspace=0.5,
    fig_kwargs={}):

    fig = plt.figure()
    fig_height_inches = (
        sum(hrows)
        )

    fig_width_inches = (
        sum(wcols)
        )

    fig=plt.figure(
        figsize=(fig_width_inches,fig_height_inches),
        subplotpars=matplotlib.figure.SubplotParams(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace =0,
        hspace = 0.0),
        **fig_kwargs)
    fig.set_size_inches(fig_width_inches,fig_height_inches,forward=True)

    gs = matplotlib.gridspec.GridSpec(
        len(hrows),
        len(wcols),
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=wspace,
        hspace=hspace,
        width_ratios=wcols,
        height_ratios=hrows
        )
    return fig, gs



from scipy.optimize import minimize
cmap = plt.cm.gnuplot2_r
def wgs_lsq_fun(x, 
                wgs_profile_for_fit,
                time_sample=25, 
                initiation_time=15, 
                shift_time=15, 
                num_samples=100, 
                L = 4040, 
                dx=10,
                return_fit=False,
                normalize_by_median=False,
                normalize_by_sum=True,
               ):
    DNAP_load_time ,low_temp_kb_per_min, high_temp_kb_per_min, DNAP_stall_time, fraction_pre_stalled = x       
    replisome_position = get_replisome_position_distribution(
                time_sample=time_sample, # time in mins to query DNA translocation 
                L=L, # genome length in kb
                DNAP_load_time=DNAP_load_time, # DNAP loading time constant in min
                initiation_time=initiation_time, # time allowed for DNAP loading 
                shift_time=shift_time, # time when the DNAP speed changes (e.g. due to temp shift)
                DNAP_stall_time =DNAP_stall_time, # time-constant in min for DNAP spontaneously stalling translocation 
                low_temp_kb_per_min=low_temp_kb_per_min,
                high_temp_kb_per_min=high_temp_kb_per_min,
                num_samples=num_samples, 
                dx=dx,
                fraction_pre_stalled=fraction_pre_stalled,
    )    
    
    N = 404
    wgs_counts = np.ones(N)*len(replisome_position)
    for rep in replisome_position:
        if any(np.isnan(rep)):
            continue
        else:
            right, left = rep
            wgs_counts[0:right] += 1
            wgs_counts[left:N] += 1
       
    #wgs_counts = wgs_counts/len(replisome_position)
    
    # normalize the wgs counts similarly to the data
    if normalize_by_sum:
        wgs_counts = wgs_counts/np.sum(wgs_counts)
        wgs_profile_for_fit = wgs_profile_for_fit/np.sum(wgs_profile_for_fit)
        resid = np.sum((wgs_counts-wgs_profile_for_fit)**2)
        
    elif normalize_by_median:
        wgs_counts = wgs_counts/np.median(wgs_counts)
        wgs_profile_for_fit = wgs_profile_for_fit/np.median(wgs_profile_for_fit)
        resid = np.sum((wgs_counts-wgs_profile_for_fit)**2)
        
    else:
        window = 10
        norm = np.median(wgs_counts[len(wgs_counts)//2-window:len(wgs_counts)//2+window])
        wgs_counts = wgs_counts/norm

        resid = np.sum((wgs_counts-wgs_profile_for_fit)**2)
    
    if return_fit is True:
        return wgs_counts
    
    return resid

def wgs_lsq_fun_time_course(x, 
                wgs_profile_for_fit_dict,
                time_samples,
                time_sample_names,
                initiation_time=15, 
                shift_time=15, 
                num_samples=100, 
                L = 4040, 
                dx=10,
                normalize_by_median=False,
                normalize_by_sum=True,
               ):
    DNAP_load_time ,low_temp_kb_per_min, high_temp_kb_per_min, DNAP_stall_time, fraction_pre_stalled = x   
    
    resid = 0
    for time_sample, time_name in zip(time_samples, time_sample_names):
        wgs_profile_for_fit = wgs_profile_for_fit_dict[f"{time_name}"]
        
        replisome_position = get_replisome_position_distribution(
                    time_sample=time_sample, # time in mins to query DNA translocation 
                    L=L, # genome length in kb
                    DNAP_load_time=DNAP_load_time, # DNAP loading time constant in min
                    initiation_time=initiation_time, # time allowed for DNAP loading 
                    shift_time=shift_time, # time when the DNAP speed changes (e.g. due to temp shift)
                    DNAP_stall_time =DNAP_stall_time, # time-constant in min for DNAP spontaneously stalling translocation 
                    low_temp_kb_per_min=low_temp_kb_per_min,
                    high_temp_kb_per_min=high_temp_kb_per_min,
                    num_samples=num_samples, 
                    dx=dx,
                    fraction_pre_stalled=fraction_pre_stalled,
        )    
        N = 404
        wgs_counts = np.ones(N)*len(replisome_position)
        for rep in replisome_position:
            if any(np.isnan(rep)):
                continue
            else:
                right, left = rep
                wgs_counts[0:right] += 1
                wgs_counts[left:N] += 1
        
        if normalize_by_sum:
            wgs_counts = wgs_counts/np.sum(wgs_counts)
            wgs_profile_for_fit = wgs_profile_for_fit/np.sum(wgs_profile_for_fit)
            resid += np.sum((wgs_counts-wgs_profile_for_fit)**2)  
            
        elif normalize_by_median:
            wgs_counts = wgs_counts/np.median(wgs_counts)
            wgs_profile_for_fit = wgs_profile_for_fit/np.median(wgs_profile_for_fit)
            resid += np.sum((wgs_counts-wgs_profile_for_fit)**2)

        else:
            window = 10
            norm = np.median(wgs_counts[len(wgs_counts)//2-window:len(wgs_counts)//2+window])
            wgs_counts = wgs_counts/norm

            resid += np.sum((wgs_counts-wgs_profile_for_fit)**2)
   
    return resid


def create_best_fit_table(x,    
                        time_samples,
                        nominal_time_samples=None,
                        initiation_time=15, 
                        shift_time=15, 
                        num_samples=100, 
                        L = 4040, 
                        dx=10,
                         ):
    DNAP_load_time ,low_temp_kb_per_min, high_temp_kb_per_min, DNAP_stall_time, fraction_pre_stalled = x   
    if nominal_time_samples is None:
        nominal_time_samples = time_samples
        
    params_dict = {
           f'DNAP_load_time_constant':[np.round(DNAP_load_time,2),'1/min', 'fit'],
           f'low_temp_DNAP_rate':[np.round(low_temp_kb_per_min,2), 'kb/min', 'fit'],    
           f'high_temp_DNAP_rate':[np.round(high_temp_kb_per_min,2), 'kb/min', 'fit'],  
           f'DNAP_stall_time_constant':[np.round(DNAP_stall_time,2), '1/min', 'fit'],   
           f'fraction_pre_stalled_DNAP':[np.round(fraction_pre_stalled,2), '', 'fit'],    
           f'num_chromosomes_for_fit':[int(num_samples**2*(1+fraction_pre_stalled)),'','fixed'],  
           f"bin_size_for_fit":[dx, 'kb', 'fixed'],         
           f"chromosome_length_in_bins":[L//dx, 'kb', 'fixed'],         
           f"time_for_DNAP_loading":[initiation_time, 'kb/min', 'fixed'], 
           f"time_at_low_temperature":[shift_time, 'min', 'fixed'],  
           f"time_to_stop_DNAP":[time_samples, 'min', 'fixed'],
           f"time_to_sample_WGS_data":[nominal_time_samples, 'min', 'fixed'],        
    }        
        

    # save simulation parameters
    params_df = pd.DataFrame(params_dict,index=['Values','Units','Fit/Fixed'])
    #print(params_df.T.to_markdown())
    plt.figure(dpi=130)
    ax = plt.subplot(111, frame_on=False)  # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    params_df.style.set_properties(**{'text-align': 'left'})
    pd.plotting.table(ax, params_df.T, loc="center")  # where df is your data frame
    plt.tight_layout()
    #plt.show()    
    return params_df 
