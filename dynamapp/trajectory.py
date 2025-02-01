"""
Trajectory Module
====================
"""
import os 
import numpy as np 
import logging 
import jax.numpy as jnp

logger = logging.getLogger(__name__)

class Trajectory:
    """
    Base class for general tarjectory motion generation.
    Args:
        - ndof - model degree of freedom
        - sampling - sampling time-genration frequancy
        - nbWaypoints - number of genated pointed of the trakejctory  
    """
    def __init__(self, ndof=7, sampling = 1000, ti=0, tf=1000) -> None:
        self.ndof = ndof
        self.sampling = sampling
        self.ti = ti
        self.tf = tf
        
    def set(self, time, Q, Qp, Qpp)->None:
        mq, nq = Q.shape
        mqp,nqp = Qp.shape
        mqpp,nqpp = Qpp.shape
        if (mq != mqp) or (mqp != mqpp) or (mq != mqpp) or \
            (nq != nqp) or (nqp != nqpp) or (nq != nqpp):
            logger.error('trajectory engine: incorrect data dimensions!')
     

class Fourier:
    """
    Base class for peroidic trajectories generation.
    
    Ref: 
        Fourier-based optimal excitation trajectories for the dynamic identification of robots
        Kyung.Jo Park - Robotica - 2006. 
    """
    def __init__(self,trajectory_params:dict) -> None:
        self.trajectory_params = trajectory_params
        
    def compute_state(self, t:float=0,q0=None, qp0=None, qpp0=None):
        """ 
        Computes the trajectory states at time date t 
        """
        pulsation = 2*np.pi*self.trajectory_params['frequancy']
        nbterms = self.trajectory_params['nbfourierterms']
        ndof = self.trajectory_params['ndof']
        q = np.zeros(ndof) if q0 is None else np.array(q0)
        qp = np.zeros(ndof) if qp0 is None else np.array(qp0)
        qpp = np.zeros(ndof) if qpp0 is None else np.array(qpp0)

        if ndof != len(q):
            logger.error('trajectory Generation Engine :inconsistency in ndof!')
        for i in range(ndof):
            for j in range(1, nbterms + 1):
                Aij = self.trajectory_params['Aij'][i][j - 1]
                Bij = self.trajectory_params['Bij'][i][j - 1]
                Cojt = np.cos(pulsation * j * t)
                Sojt = np.sin(pulsation * j * t)
                q[i] += Aij / (pulsation * j) * Sojt - Bij / (pulsation * j) * Cojt
                qp[i] += Aij * Cojt + Bij * Sojt
                qpp[i] += Bij * j * Cojt - Aij * j * Sojt
            qpp[i] = pulsation * qpp[i]
        return q, qp, qpp
    
    def compute_full_trajectory(self, ti: float, tf: float,q0=None, qp0=None, qpp0=None):
        """
        Computes the full trajectory data between ti and tf 
        Args:
            - 
        Returns:
            - q, qp, qpp
        """
        ndof = self.trajectory_params['ndof']
        nb_traj_samples = self.trajectory_params['samples']
        time = np.linspace(ti, tf, nb_traj_samples)
        q = np.zeros((nb_traj_samples,ndof))
        qp = np.zeros((nb_traj_samples,ndof))
        qpp = np.zeros((nb_traj_samples,ndof))
        for i in range(nb_traj_samples):
            q[i,:], qp[i,:], qpp[i,:] = self.compute_state(time[i],q0,qp0,qpp0)
            
        return q, qp, qpp

    
    def compute_error(self,x,tspan,new_traj_params=None,q0=None,qp0=None,\
                               qpp0=None,verbose=False):
      
        if not(new_traj_params is None):
            k = self.trajectory_params['ndof'] * self.trajectory_params['nbfourierterms']
            self.trajectory_params['Aij'] = np.reshape(new_traj_params[0:k],(-1,5))
            self.trajectory_params['Bij'] = np.reshape(new_traj_params[k:2*k],(-1,5))
        err = self.computeDifferentiationError(0,tspan,x,q0,qp0,qpp0)
        if verbose:
            print( f"RMSE = {err:.5f}")
         
        return err
    
    def _computeTrajectoryConstraints(self,ti,tf,qmax,qmin,qpmax,qpmin,qppmin,qppmax,\
        q0=None, qp0= None, qpp0=None):
        """ 
        Computes the trajectory with taking constraintes into account 
        """
        ndof = self.trajectory_params['ndof']
        nb_traj_samples = self.trajectory_params['samples']
        time = np.linspace(ti, tf, nb_traj_samples)
        q = np.zeros((nb_traj_samples,ndof))
        qp = np.zeros((nb_traj_samples,ndof))
        qpp = np.zeros((nb_traj_samples,ndof))
        for i in range(nb_traj_samples):
            q[i,:], qp[i,:], qpp[i,:] = self.computeTrajectoryState(time[i],q0,qp0,qpp0)
            q[i,:] = np.clip(q[i,:],qmin,qmax)
            qp[i,:] = np.clip(qp[i,:],qpmin,qpmax)
            qpp[i,:] = np.clip(qpp[i,:],qppmin,qppmax)

        return q, qp ,qpp
    
class Trapezoidal:
    """
    Base class for trapezoidal trajectories generation 
    Args:
        - njoints     number of joints
        - nwaypoints  number of waypoints
        
                    acceleration values
                    accelerated durations
                    vel constant durations
                    runtime
    Output:
                    q, qd, qdd
    """

    def __init__(self, njoints, nwaypoints, acc, delta_t1, delta_t2, Nf):
        self.njoints = njoints

        self.q0 = np.random.uniform(-np.pi / 2, np.pi / 2, size=(njoints,))
        self.qd0 = np.zeros(njoints)
        self.qdd0 = acc[0, :]

        self.Kq = []
        self.Kv = []
        self.Ka = []
        # (nwaypoints -1 x njoints), matrix of acceleratiion on 1st accelearated
        # duration
        self.acc = acc
        # (nwaypoints -1 x njoints)1st accelerated duration
        self.delta_t1 = delta_t1
        # (nwaypoints -1 x njoints)constant vel duration
        self.delta_t2 = delta_t2

        self.nwaypoints = nwaypoints
        # (nwaypoints - 1x njoints)a list of runtime between 2 consecutive waypoints
        self.Nf = Nf
        self.ts = 0.01

    def TrapezoidalGenerator(self):
        self.initConfig()
        if self.nwaypoints == 1:
            print("Number of waypoints needs to be more 1!")
        else:
            for i in range(self.nwaypoints - 1):
                for j in range(self.njoints):
                    # at one joint between 2 waypoints
                    q_, qd_, qdd_ = self.trapTraj_PTP(
                        self.acc[i, j],
                        self.q[j][-1],
                        self.delta_t1[i, j],
                        self.delta_t2[i, j],
                        self.Nf[i],
                    )
                    self.q[j] = np.append(self.q[j], q_)
                    self.qd[j] = np.append(self.qd[j], qd_)
                    self.qdd[j] = np.append(self.qdd[j], qdd_)
        # self.plotTraj()
        return self.q, self.qd, self.qdd

    def initConfig(self):
        self.q = []
        self.qd = []
        self.qdd = []
        for i in range(self.njoints):
            self.q.append(np.array([self.q0[i]]))
            self.qd.append(np.array([self.qd0[i]]))
            self.qdd.append(np.array([self.qdd0[i]]))

    def trapTraj_PTP(self, a1, q0, n1, n2, N):
        ts = self.ts
        q_ = np.array([q0])
        qd_ = np.array([0])
        qdd_ = np.array([a1])
        # acceleration on 2nd accelarated duration to ensure vel(end) = 0
        a3 = -a1 * n1 / (N - n1 - n2)
        for i in range(1, N):
            if i < n1:
                qdd_ = np.append(qdd_, a1)
                qd_ = np.append(qd_, qd_[i - 1] + qdd_[i - 1] * ts)
                q_ = np.append(q_, q_[i - 1] + qd_[i - 1] * ts)
            elif i >= n1 and i < (n1 + n2):
                qdd_ = np.append(qdd_, 0)
                qd_ = np.append(qd_, qd_[i - 1] + qdd_[i - 1] * ts)
                q_ = np.append(q_, q_[i - 1] + qd_[i - 1] * ts)
            else:
                qdd_ = np.append(qdd_, a3)
                qd_ = np.append(qd_, qd_[i - 1] + qdd_[i - 1] * ts)
                q_ = np.append(q_, q_[i - 1] + qd_[i - 1] * ts)
        return q_, qd_, qdd_

 


class Spline:
    """
    Base Class for spline trajectories generation
    """
    def __init__(self, trajectory_params) -> None:
        self.trajectory_params = trajectory_params
    
    def computeTrajectoryState(self, t, Q0=None) -> np.ndarray:
        """Computes the trajectory states at time t."""
        if Q0 is None:
            Q0 = self.trajectory_params['Q0']
        cs = CubicSpline(self.trajectory_params['time_points'], Q0)
        states = cs(t)
        return states
    
    def computeTrajectoryIdentifiability(self):
        """Evaluates the regression criteria ε(q, qp, qpp, x)."""
        epsilon = 0.0
        logger.info("Computing trajectory identifiability criteria.")
        
        return epsilon
    
    def computeFullTrajectory(self,ti:float,tf:float,q0=None,qp0=None,qpp0=None):
        """Computes the full trajectory between ti and tf."""
        t = np.linspace(ti, tf, 100)  
        q = self.computeTrajectoryState(t, q0)
        qp = np.gradient(q,t)  
        qpp = np.gradient(qp,t)  
        logger.info("Full trajectory computed.")
        return q,qp,qpp
        
    def computeTrajectoryConstraints(self,qmax,qmin,qpmax,qpmin,qppmin,qppmax,ti,tf,\
        q0=None,qp0=None,qpp0=None):
        """Ensures trajectory meets specified constraints."""
        q, qp, qpp = self.computeFullTrajectory(ti, tf, q0, qp0, qpp0)
        is_within_constraints = (
            np.all(q >= qmin) and np.all(q <= qmax) and
            np.all(qp >= qpmin) and np.all(qp <= qpmax) and
            np.all(qpp >= qppmin) and np.all(qpp <= qppmax)
        )
        logger.info(f"Trajectory constraints check: {is_within_constraints}")
        
        return is_within_constraints
    
