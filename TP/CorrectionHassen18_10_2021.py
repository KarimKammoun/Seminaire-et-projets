#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:43:41 2020

@author: drira
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scipy.io
import numpy as np
import scipy
from numpy import linalg as LA
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from math import atan
import math
from scipy.linalg import norm

def projectFramePreshape(frame):
	#Tapez votre code ici
    #Puis on centre on soustrayant la moyenne des points
	meanX = np.mean([a for [a,b] in frame])
	meanY = np.mean([b for [a,b] in frame])
	meanPoint = [meanX, meanY]
	result = np.subtract(frame, meanPoint)
	#On divise d'abord la frame par la norme de Frobenius
	result = np.divide(result, LA.norm(result, 'fro'))
	return result

def DProduitScalaireFrobinus(X,Y):
    #Tapez votre code ici
    X=projectFramePreshape(X)
    Y=projectFramePreshape(Y)
    mat = X.dot(np.transpose(Y))
    CosAngle =  np.trace(mat)
    if CosAngle>1:
        CosAngle=1   
    return math.acos(CosAngle)

def dtw(s, t):
    n=s.shape[1]
    m=t.shape[1]
    #n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            #cost = abs(s[i-1] - t[j-1])
            cost = DProduitScalaireFrobinus(s[:,i-1,:],t[:,j-1,:])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix  


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform



def DistanceSequences(X,Y):
    i=0
    NbFramesX = X.shape[1]
    NbFramesY = Y.shape[1]
    NbFrames = min(NbFramesX,NbFramesY)
    dist = [None] * NbFrames
    while i < NbFrames:
        FrameCouranteX = X[:,i,:]
        FrameCouranteXPreshape = projectFramePreshape(FrameCouranteX)
        FrameCouranteY = Y[:,i,:]
        FrameCouranteYPreshape = projectFramePreshape(FrameCouranteY)
        dist[i] = DProduitScalaireFrobinus(FrameCouranteXPreshape,FrameCouranteYPreshape)
        i=i+1
    return dist


matrice = scipy.io.loadmat('sequences_ckp.mat')
TabSequences = matrice['sequences'][0]

labels = scipy.io.loadmat('labels_ckp.mat')
TabLabels = labels['labels'][0]

sujetID = scipy.io.loadmat('subject_labels_ckp.mat')
TabSubjects = sujetID['subject_labels']
TabSubjects = TabSubjects[:,0]


Sequence326 = TabSequences[326]   # sequence n 326
Frame = Sequence326[:,0,:]
FramePreshape = projectFramePreshape(Frame)

Frame1 = Sequence326[:,11,:]
Frame1Preshape = projectFramePreshape(Frame1)


a = np.array([[0, -1],[ 1, 0]], float)
Frame1Preshape90 = Frame1Preshape.dot(a)


# Calculate transform via procrustes...
FrameXX, Frame1PreshapeAligned, disparity = procrustes(Frame1Preshape90, Frame1Preshape)
#d,Frame1PreshapeAligned,Tform = procrustes(Frame1Preshape90,Frame1Preshape)



#visage1 = Frame326[:,0,:]
#plt.plot(FramePreshape[:,0],FramePreshape[:,1],'x')
#plt.savefig("FramePreshape.png")

plt.plot(Frame1Preshape90[:,0],Frame1Preshape90[:,1],'x',color='red')
plt.plot(Frame1Preshape[:,0],Frame1Preshape[:,1],'x',color='blue')
plt.plot(Frame1PreshapeAligned[:,0],Frame1PreshapeAligned[:,1],'x',color='green')

plt.savefig("FramePreshapeAligned.png")

plt.show()

#plt.show()

