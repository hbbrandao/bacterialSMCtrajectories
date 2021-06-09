#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=True

import numpy as np
cimport numpy as np 
import cython
cimport cython 


cdef extern from "<stdlib.h>":
    double drand48()   

cdef cython.double randnum():
    return drand48()


cdef class smcTranslocatorCircular_DissociatesOrBypasses(object):
    cdef int N
    cdef int M
    cdef cython.double [:] emission
    cdef cython.double [:] stallLeft
    cdef cython.double [:] stallRight
    cdef cython.double [:] stallFalloff
    cdef cython.double [:] falloff
    cdef cython.double [:] pauseL
    cdef cython.double [:] pauseR
    cdef cython.double [:] cumEmission
    cdef cython.long [:] SMCs1
    cdef cython.long [:] SMCs2
    cdef cython.long [:] stalled1 
    cdef cython.long [:] stalled2
    cdef cython.long [:] occupied 
    
    #cdef cython.double asymmetricDissociation
    cdef int maxss
    cdef int curss
    cdef cython.double knockOffProb_TerToOri
    cdef cython.double knockOffProb_OriToTer
    cdef cython.double kBypass
    cdef cython.long [:] ssarray  
 
    
    def __init__(self, emissionProb, deathProb, stallProbLeft, stallProbRight, pauseProbL, pauseProbR, stallFalloffProb, kkOriToTer_kkTerToOri_kBypass,  numSmc):
     
        
        self.N = len(emissionProb)
        self.M = numSmc
        self.emission = emissionProb
        self.stallLeft = stallProbLeft
        self.stallRight = stallProbRight
        self.falloff = deathProb
        self.pauseL = pauseProbL
        self.pauseR = pauseProbR
        cumem = np.cumsum(emissionProb)
        cumem = cumem / float(cumem[len(cumem)-1])
        self.cumEmission = np.array(cumem, np.double)
        self.SMCs1 = np.zeros((self.M), int)
        self.SMCs2 = np.zeros((self.M), int)
        self.stalled1 = np.zeros(self.M, int)
        self.stalled2 = np.zeros(self.M, int)
        self.occupied = np.zeros(self.N, int)
        self.stallFalloff = stallFalloffProb
        self.knockOffProb_OriToTer = kkOriToTer_kkTerToOri_kBypass[0]
        self.knockOffProb_TerToOri = kkOriToTer_kkTerToOri_kBypass[1] # rate of facilitated dissociation
        self.kBypass = kkOriToTer_kkTerToOri_kBypass[2]
        
        self.maxss = 1000000
        self.curss = 99999999

        for ind in xrange(self.M):
            self.birth(ind)


    cdef birth(self, cython.int ind):
        cdef int pos,i 
  
        while True:
            pos = self.getss()
            if pos > self.N - 1: 
                print "bad value", pos, self.cumEmission[len(self.cumEmission)-1]
                continue 
            if pos < 0: 
                print "bad value", pos, self.cumEmission[0]
                continue 
 
            
            if (self.occupied[pos] >= 1) or (self.occupied[np.mod(pos+1,self.N)] >= 1): # CHANGED THIS
                continue
            
            if (pos > (self.N - 1)):  
                continue
                
            self.SMCs1[ind] = pos
            self.SMCs2[ind] = np.mod(pos+1,self.N)
            self.occupied[pos] += 1 # CHANGED THIS
            self.occupied[np.mod(pos+1,self.N)] += 1    # CHANGED THIS   
            
            return

    cdef death(self):
        cdef int i 
        cdef double falloff1, falloff2 
        cdef double falloff 
         
        for i in xrange(self.M):
            if self.stalled1[i] == 0:
                falloff1 = self.falloff[self.SMCs1[i]]
            else: 
                falloff1 = self.stallFalloff[self.SMCs1[i]]
            if self.stalled2[i] == 0:
                falloff2 = self.falloff[self.SMCs2[i]]
            else:
                falloff2 = self.stallFalloff[self.SMCs2[i]]              
            
            falloff = max(falloff1, falloff2)
            if randnum() < falloff:                 
                self.occupied[self.SMCs1[i]] -= 1  # CHANGED THIS
                self.occupied[self.SMCs2[i]] -= 1  # CHANGED THIS
                self.stalled1[i] = 0
                self.stalled2[i] = 0
                self.birth(i)
    
    cdef int getss(self):
    
        if self.curss >= self.maxss - 1:
            foundArray = np.array(np.searchsorted(self.cumEmission, np.random.random(self.maxss)), dtype = np.long)
            self.ssarray = foundArray
            #print np.array(self.ssarray).min(), np.array(self.ssarray).max()
            self.curss = -1
        
        self.curss += 1         
        return self.ssarray[self.curss]
        
        

    cdef step(self):
        cdef int i 
        cdef double pause
        cdef double stall1, stall2 
        cdef int cur1
        cdef int cur2 
        cdef int newCur1
        cdef int newCur2
        for i in range(self.M):            
            stall1 = self.stallLeft[self.SMCs1[i]]
            stall2 = self.stallRight[self.SMCs2[i]]
                                                      
            # update positions             
            cur1 = self.SMCs1[i]
            cur2 = self.SMCs2[i]
            
            # new positions with the periodic boundaries
            newCur1 = np.mod(cur1 - 1,self.N) # left 
            newCur2 = np.mod(cur2 + 1,self.N) # right 
            
            # reset "is stalled" -> this actually means "will dissociate from collision"
            self.stalled1[i] = 0
            self.stalled2[i] = 0               
            
            # move each side only if the other side is "free" or stall/knock off
            if self.stalled1[i] == 0: 
                # bypass
                pause1 = self.pauseL[self.SMCs1[i]]
                if randnum() > pause1: 
                    if (self.occupied[newCur1] == 0):
                        # take a normal step 
                        self.occupied[newCur1] += 1
                        self.occupied[cur1] -= 1
                        self.SMCs1[i] = newCur1  
                    else:
                        rateSum = self.knockOffProb_OriToTer+self.kBypass
                        if randnum() <= rateSum: 
                            if randnum() <= self.knockOffProb_OriToTer/rateSum:
                                # stall and maybe dissociate
                                self.stalled1[i] = 1
                            else:
                                self.occupied[newCur1] += 1
                                self.occupied[cur1] -= 1
                                self.SMCs1[i] = newCur1                              
                        
            if self.stalled2[i] == 0:  
                pause2 = self.pauseR[self.SMCs2[i]]
                if randnum() > pause2:                 
                    if (self.occupied[newCur2] == 0) :        
                        # take a normal step 
                        self.occupied[newCur2] += 1
                        self.occupied[cur2] -= 1
                        self.SMCs2[i] = newCur2
                    else:
                        rateSum = self.knockOffProb_TerToOri+self.kBypass
                        if randnum() <= rateSum:                     
                            if randnum() <= self.knockOffProb_TerToOri/rateSum:
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
        cdef int i 
        for i in xrange(N):
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



