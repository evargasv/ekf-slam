#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
import numpy as np
from functions import angle_wrap, comp, compInv
import scipy.linalg
import rospy
import ipdb
#============================================================================
class EKF_SLAM(object):
    '''
    Class to hold the whole EKF-SLAM.
    '''

    #========================================================================
    def __init__(self, x0, y0, theta0, odom_lin_sigma,
                 odom_ang_sigma, meas_rng_noise, meas_ang_noise):
        '''
        Initializes the ekf filter
        room_map : an array of lines in the form [x1 y1 x2 y2]
        num      : number of particles to use
        odom_lin_sigma: odometry linear noise
        odom_ang_sigma: odometry angular noise
        meas_rng_noise: measurement linear noise
        meas_ang_noise: measurement angular noise
        '''

        # Copy parameters
        self.odom_lin_sigma = odom_lin_sigma
        self.odom_ang_sigma = odom_ang_sigma
        self.meas_rng_noise = meas_rng_noise
        self.meas_ang_noise = meas_ang_noise
        self.chi_thres = 0.1026

        # Odometry uncertainty
        self.Qk = np.array([[ self.odom_lin_sigma**2, 0, 0],\
                            [ 0, self.odom_lin_sigma**2, 0 ],\
                            [ 0, 0, self.odom_ang_sigma**2]])

        # Measurements uncertainty
        self.Rk=np.eye(2)
        self.Rk[0,0] = self.meas_rng_noise
        self.Rk[1,1] = self.meas_ang_noise

        # State vector initialization
        self.x_B_1 = np.array([x0,y0,theta0]) # Position
        self.P_B_1 = np.zeros((3,3)) # Uncertainty

        # Initialize buffer for forcing observing n times a feature before
        # adding it to the map
        self.featureObservedN = np.array([])
        self.min_observations = 0

    #========================================================================
    def get_number_of_features_in_map(self):
        '''
        returns the number of features in the map
        '''
        return (self.x_B_1.size-3)/2

    #========================================================================
    def get_polar_line(self, line, odom):
        '''
        Transforms a line from [x1 y1 x2 y2] from the world frame to the
        vehicle frame using odometry [x y ang].
        Returns [range theta]
        '''
        # Line points
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        # Compute line (a, b, c) and range
        line = np.array([y1-y2, x2-x1, x1*y2-x2*y1])
        pt = np.array([odom[0], odom[1], 1])
        dist = np.dot(pt, line) / np.linalg.norm(line[:2])

        # Compute angle
        if dist < 0:
            ang = np.arctan2(line[1], line[0])
        else:
            ang = np.arctan2(-line[1], -line[0])

        # Return in the vehicle frame
        return np.array([np.abs(dist), angle_wrap(ang - odom[2])])

    #========================================================================
    def predict(self, uk):
        '''
        Predicts the position of the robot according to the previous position
        and the odometry measurements. It also updates the uncertainty of the
        position
        '''
        # - Update self.x_B_1 and self.P_B_1 using uk and self.Qk

        # number of observed features
        n = self.get_number_of_features_in_map()

        # Compute jacobians of the composition with respect to robot (A_k)
        # and odometry (W_k)
        # A_k
        A_k = np.array([[1, 0, -np.sin(self.x_B_1[2]) * uk[0] - np.cos(self.x_B_1[2]) * uk[1]],\
                       [0, 1,  np.cos(self.x_B_1[2]) * uk[0] - np.sin(self.x_B_1[2]) * uk[1]],\
                       [0, 0, 1]])
        # W_k
        W_k = np.array([[np.cos(self.x_B_1[2]), -np.sin(self.x_B_1[2]), 0],\
                       [np.sin(self.x_B_1[2]),  np.cos(self.x_B_1[2]), 0],\
                       [0,0,1]])

        # Prepare the F_k and G_k matrix for the new uncertainty computation
        # F_k
        F_k = scipy.linalg.block_diag(A_k, np.eye(2*n))
        # G_k
        G_k = np.vstack((W_k,np.zeros((2*n,3))))

        # Update the class variables
        # Compound robot with odometry
        self.x_B_1[0] += ( np.cos(self.x_B_1[2]) * uk[0] ) - ( np.sin(self.x_B_1[2]) * uk[1] )
        self.x_B_1[1] += ( np.sin(self.x_B_1[2]) * uk[0] ) + ( np.cos(self.x_B_1[2]) * uk[1] )
        self.x_B_1[2] = angle_wrap( self.x_B_1[2] + uk[2] )
        # Compute uncertainty
        #ipdb.set_trace()
        self.P_B_1 = ( F_k * np.mat(self.P_B_1) * F_k.T ) + ( G_k * np.mat(self.Qk) * G_k.T )

    #========================================================================

    def data_association(self, lines):
        '''
        Implements ICCN for each feature of the scan.
        Innovk_List -> matrix containing the innovation for each associated
                       feature
        H_k_List -> matrix of the jacobians.
        S_f_List -> matrix of the different S_f for each feature associated
        Rk_List -> matrix of the noise of each measurement
        '''
        # for each sensed line do:
        #   1- Transform the sened line to polar
        #   2- for each feature of the map (in the state vector) compute the
        #      mahalanobis distance
        #   3- Data association

        # number of features in the map
        n = self.get_number_of_features_in_map()

        # Init variable
        Innovk_List   = np.zeros((0,1))
        H_k_List      = np.zeros((0,3+2*n))
        S_f_List      = np.zeros((0,2))
        Rk_List       = np.zeros((0,2*n))
        idx_not_associated = np.zeros((0,1))

        # 1- Sensed lines
        for i in xrange( lines.shape[0] ):

            # transform line in polar coordinates
            polar_line = self.get_polar_line( lines[i], np.array([0,0,0]) )

            # initialize minimums
            v_min = np.zeros((2, 1))
            H_min = np.zeros((2, 3+2*n))
            S_min = np.zeros((2 , 2))
            D_min = float("inf")

            # number of features in the map
            for j in xrange( n ):

                # initialize arrays
                v = np.zeros((2, 1))
                H = np.zeros((2, 3+2*n))
                S = np.zeros((2, 2))
                D = 0

                # 2- Mahalanobis distance
                D,v,h,H,S = self.lineDist( polar_line, 3+2*j)

                # Validate if the Mahalobis distance is the minimum
                if D < D_min:
                    v_min = v
                    H_min = H
                    S_min = S
                    D_min = D

            # Validate Mahalanobis distance threshold
            if D_min < self.chi_thres:
                # 3- Data association
                Innovk_List = np.vstack(( Innovk_List, v_min ))
                H_k_List = np.vstack (( H_k_List, H_min ))
                S_f_List = np.vstack(( S_f_List, S_min ))
            else:
                # add idx to the idx_not_associated, which will be added in the state aumentation
                idx_not_associated = np.append( idx_not_associated, i )
        # Rk
        Rk_List = self.calculate_Rk( S_f_List.shape[0] / 2 )

        return Innovk_List, H_k_List, S_f_List, Rk_List, idx_not_associated

    #===========================================================================
    def calculate_Rk( self, n ):

        Rk = np.zeros((2*n, 2*n))

        for i in xrange(n):
            Rk[i*2:i*2+2,i*2:i*2+2] = self.Rk

        return Rk

    #========================================================================
    def update_position(self, Innovk_List, H_k_List, S_f_List , Rk_List) :
        '''
        Updates the position of the robot according to the given the position
        and the data association parameters.
        Returns state vector and uncertainty.
        '''
        if Innovk_List.shape[0]<1:
            return

        # number of features in the map
        n = self.get_number_of_features_in_map()

        # Kalman Gain
        Sk = H_k_List * np.mat(self.P_B_1) * H_k_List.T + Rk_List
        Kk = np.mat(self.P_B_1) * H_k_List.T * np.linalg.inv(Sk)

        # Update Position
        self.x_B_1 += ( Kk * Innovk_List ).A1

        # Update Uncertainty
        IKH = np.eye(3+2*n) - Kk * H_k_List
        self.P_B_1 = ( IKH * np.mat(self.P_B_1) * IKH.T ) + Kk * Rk_List * Kk.T
        self.P_B_1 = np.array(self.P_B_1)


    #========================================================================
    def state_augmentation(self, lines, idx):
        '''
        given the whole set of lines read by the kineckt sensor and the
        indexes that have not been associated augment the state vector to
        introduce the new features
        '''
        # If no features to add to the map exit function
        if idx.size<1:
            return

        # number of features in the map
        n = self.get_number_of_features_in_map()
        # number of features to add to the map
        f = idx.size

        # initialize Fk and Gk
        F_k = np.vstack(( scipy.linalg.block_diag(np.eye(3+2*n)), np.zeros(((2*f),3+(2*n))) ))
        G_k = np.zeros((3+(2*n)+(2*f),2))

        # initialize x_F, the augmented vector of features
        x_F = np.zeros((2*f))

        # stack the features to be added
        for i in xrange(idx.shape[0]):
            # convert line to polar coordinates
            #ipdb.set_trace()
            polar_line = self.get_polar_line(lines[idx[i],:], np.array([0,0,0]))
            # transform the polar line to world frame
            x_F[2*i:2*i+2], J_1, J_2 = self.tfPolarLine(self.x_B_1[0:3],polar_line)
            # set values in F_k and G_k
            idx_row = [(3+2*n)+2*i,(3+2*n)+2*i+1]
            F_k [idx_row,0:3] = J_1
            G_k [idx_row,0:2] = J_2

        # augment the state vector
        self.x_B_1 = np.hstack(( self.x_B_1, x_F ))
        # calculate features uncertainty
        self.P_B_1 = ( F_k * np.mat(self.P_B_1) * F_k.T ) + ( G_k * np.mat(self.Rk) * G_k.T )

    #========================================================================
    def tfPolarLine(self,tf,line):
        '''
        Transforms a polar line in the robot frame to a polar line in the
        world frame
        '''

        # Decompose transformation
        x_x = tf[0]
        x_y = tf[1]
        x_ang = tf[2]

        # Compute the new phi
        phi = angle_wrap(line[1] + x_ang)

        # Auxiliar computations
        sqrt2 = x_x**2+x_y**2
        sqrt = np.sqrt(sqrt2)
        atan2 = np.arctan2(x_y,x_x)
        sin = np.sin(atan2 - phi)
        cos = np.cos(atan2 - phi)

        # Compute the new rho
        rho = line[0] + sqrt* cos

        # Allocate jacobians
        H_tf = np.zeros((2,3))
        H_line = np.eye(2)

        # Evaluate jacobian respect to transformation
        H_tf[0,0] = ( x_x / sqrt ) * cos + ( x_y / sqrt ) * sin
        H_tf[0,1] = ( x_y / sqrt ) * cos - ( x_x / sqrt ) * sin
        H_tf[0,2] = sqrt * sin
        H_tf[1,2] = 1

        # Evaluate jacobian respect to line
        H_line[0,0] = 1
        H_line[0,1] = sqrt * sin
        H_line[1,1] = 1

        return np.array([rho,phi]), H_tf, H_line

    #========================================================================
    def lineDist(self,z,idx):
        '''
        Given a line and an index of the state vector it computes the
        distance between both lines
        '''
        # Transform the map line into robot frame and compute jacobians

        # map line
        map_line = self.x_B_1[idx:idx+2]
        # inverse transformation
        x_R_B, J = compInv( self.x_B_1[0:3] )
        # convert to polar line and compute jacobians
        h, H_position, H_line = self.tfPolarLine( x_R_B, map_line )

        # Allocate overall jacobian
        H = np.zeros(( 2, self.x_B_1.size ))

        # Concatenate position jacobians and place them into the position
        H[ :, 0:3 ] = H_position * np.mat( J )

        # Place the position of the jacobian with respect to the line in its
        # position
        H[ :, idx:idx+2 ] = H_line

        # Calculate innovation
        v = np.mat( z - h ).T

        # Calculate innovation uncertainty
        S = self.Rk + H * np.mat( self.P_B_1 ) * H.T

        # Calculate mahalanobis distance
        D = v.T * np.linalg.inv(S) * v

        return D,v,h,H,S
