import logging
import jax
from jax import jit
import jax.numpy as jnp
from .viscoelastic import *

logger = logging.getLogger(__name__)

class Model():
    """
    Base class definition for multijoint models.

    Args:
        - dhparams : model DH parameters.
        - Imats : links inertia tensors, list of JAX arrys
        - q (np.ndarray): Joints position vector.
        - v (np.ndarray): Joints velocity vector.
        - a (np.ndarray): Joints acceleration vector.
        - dampings : joints damping coefficents list
        - gravity : gravity vector applied to the system
        
     Examples:
        >>> model = Model(Imats, dhparams, -9.80, dampings)
    """
    def __init__(self, Imats:list, 
                dhparams:list, 
                gravity = -9.81, 
                dampings:list = None)->None:
        
        self.Imats = Imats
        self.dampings = dampings
        self.dhparams = dhparams
        self.gravity = gravity  
        assert len(Imats) == len(dhparams)
        if dampings is not None:
            assert len(dampings) == len(Imats)
        self.ndof = len(Imats)
        self.q = jnp.zeros(self.ndof) 
        self.v = jnp.zeros(self.ndof) 
        self.a = jnp.zeros(self.ndof)
    
    def _rnea(self, q:jnp.ndarray, 
            qp:jnp.ndarray,
            qpp:jnp.ndarray=None
            )->tuple:
        """
        Recursive Newton-Euler Algorithm (RNEA) implementation For the inverse Dynamic Model in JAX. 
         
        Args:
            - q : joints position vector
            - qp : joints velocity vector
            - qpp : joints acceleration vector
        Retruns:
            - c: coriolis terms and other forces potentially be applied to the system. 
            - v: velocity of each joint in world base coordinates.
            - a: acceleration of each joint in world base coordinates.
            - f: forces that joints must apply to produce trajectory
        """
        n = len(qp)
        v = jnp.zeros((6,n))
        a = jnp.zeros((6,n))
        f = jnp.zeros((6,n))
        c = jnp.zeros(n)
        gravity_vec = jnp.zeros((6))
        gravity_vec = gravity_vec.at[5].set(-self.gravity)
        parent_ids = [-1] * n
        
        for i in range(n):
            parent_i = parent_ids[i]
            Xmat = self._transform(i, q[i])
            S = self._screw(i)
            
            if parent_i == -1:
                a = a.at[:,i].set(jnp.matmul(Xmat, gravity_vec))
            else:
                v = v.at[:,i].set(jnp.matmul(Xmat,v[:,parent_i]))
                a = a.at[:,i].set(jnp.matmul(Xmat,a[:,parent_i]))
                
            if qpp is not None:
                v_ = S*qpp[i] 
                v_up = v[:,i] + v_
                v = v.at[:,i].set(v_up)
               
            a_up = a[:,i] + self._mxS(S, v[:,i], qp[i])    
            a = a.at[:,i].set(a_up) 
            
            if qpp is not None:
                a_upp = a[:,i]+ S*qpp[i]
                a = a.at[:,i].set(a_upp)
            f = f.at[:,i].set(jnp.matmul(self.Imats[i],a[:,i]) + self._vxIv(v[:,i],self.Imats[i]))
        
        for i in range(n-1,-1,-1):
            S = self._screw(i)
            c = c.at[i].set(jnp.matmul(jnp.transpose(S), f[:,i]))
            parent_i = parent_ids[i]
            if parent_i != -1:
                Xmat = self._transform(i, q[i])
                temp = jnp.matmul(jnp.transpose(Xmat),f[:,i])
                f_up = f[:,parent_i] + temp.flatten()
                f = f.at[:,parent_i].set(f_up)

        if self.dampings is not None:
            for k in range(n):
                c_up = c.at[k] + self.dampings(k) * qp[k]
                c = c.at[k].set(c_up) 
        
        return c, v, a, f
    
    def damping_tensor(self) -> jnp.ndarray:
        """
        Get the diagonal velocity damping matrix of size (ndof, ndof).
        
        Returns:
            jnp.ndarray: A diagonal matrix with damping values.
                        If no damping values are given, returns a zero matrix.
        """
        dampings = jnp.array(self.dampings) if self.dampings is not None else jnp.zeros(self.dampings.length())
        
        return jnp.diag(dampings)
    
    @staticmethod 
    @jit
    def _vxIv(vec:jnp.ndarray, Imat:jnp.ndarray) -> jnp.ndarray:
        """
        Computes the cross-product of a spatial velocity vector with the 
        inertia matrix product (I * v).

        Args:
            vec: (6,) Spatial velocity vector.
            Imat: (6,6) Inertia matrix.

        Returns:
            vecXIvec: (6,) Resultant vector after cross-product.
        """
        assert vec.shape == (6,), f"Expected vector shape (6,), but got {vec.shape}"
        assert Imat.shape == (6, 6), f"Expected Inertia Matrix shape (6, 6), but got {Imat.shape}"
        temp = jnp.matmul(Imat, vec) 
        vecXIvec = jnp.zeros(6)
        vecXIvec = vecXIvec.at[:3].set(jnp.cross(vec[:3], temp[:3]) + jnp.cross(vec[3:], temp[3:]))
        vecXIvec = vecXIvec.at[3:].set(jnp.cross(vec[:3], temp[3:]))

        return vecXIvec
    
    @staticmethod
    @jit
    def _mxS(S: jnp.ndarray, vec: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
        """
        Computes the motion cross-product operation based on the screw axis S.

        Args:
            - S: (6,) Screw axis vector (indicating motion type).
            - vec: (6,) Spatial vector.
            - alpha: Scaling factor (default = 1.0).

        Returns:
            - vecX: (6,) Resultant motion cross-product vector.
        """
        vecX = jnp.zeros(6)

        vecX = vecX.at[0].set(jnp.where(S[1] == 1, -vec[2] * alpha, jnp.where(S[2] == 1, vec[1] * alpha, 0)))
        vecX = vecX.at[1].set(jnp.where(S[0] == 1, vec[2] * alpha, jnp.where(S[2] == 1, -vec[0] * alpha, 0)))
        vecX = vecX.at[2].set(jnp.where(S[0] == 1, -vec[1] * alpha, jnp.where(S[1] == 1, vec[0] * alpha, 0)))

        vecX = vecX.at[3].set(jnp.where(S[1] == 1, -vec[5] * alpha, 
                            jnp.where(S[2] == 1, vec[4] * alpha, 
                            jnp.where(S[4] == 1, -vec[2] * alpha, 
                            jnp.where(S[5] == 1, vec[1] * alpha, 0)))))
        
        vecX = vecX.at[4].set(jnp.where(S[0] == 1, vec[5] * alpha, 
                            jnp.where(S[2] == 1, -vec[3] * alpha, 
                            jnp.where(S[3] == 1, vec[2] * alpha, 
                            jnp.where(S[5] == 1, -vec[0] * alpha, 0)))))

        vecX = vecX.at[5].set(jnp.where(S[0] == 1, -vec[4] * alpha, 
                            jnp.where(S[1] == 1, vec[3] * alpha, 
                            jnp.where(S[3] == 1, -vec[1] * alpha, 
                            jnp.where(S[4] == 1, vec[0] * alpha, 0)))))

        return vecX
    
    @staticmethod
    def _screw(i:int):
        """Retrieves the screw axis S for joint i."""
        axes = {
            0: jnp.array([0, 0, 1, 0, 0, 0]),  
            1: jnp.array([0, 1, 0, 0, 0, 0]),  
            2: jnp.array([1, 0, 0, 0, 0, 0]),  
            3: jnp.array([0, 0, 0, 1, 0, 0]),   
            4: jnp.array([0, 0, 0, 0, 1, 0]),  
            5: jnp.array([0, 0, 0, 0, 0, 1]), 
        }
        return axes.get(i, jnp.zeros(6))
    
    def _transform(self, i, qi)->jnp.ndarray:
        """Computes the transformation matrix X for joint i at configuration q_i"""
        
        theta, d, a, alpha = self.dhparams[i]
        theta += qi
        T = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta) * jnp.cos(alpha),  jnp.sin(theta) * jnp.sin(alpha), a * jnp.cos(theta)],
            [jnp.sin(theta),  jnp.cos(theta) * jnp.cos(alpha), -jnp.cos(theta) * jnp.sin(alpha), a * jnp.sin(theta)],
            [0,              jnp.sin(alpha),                  jnp.cos(alpha),                  d],
            [0,              0,                               0,                               1]
        ])
        R = T[:3, :3]  
        p = T[:3, 3]
        p_hat = jnp.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
        mat = jnp.block([
            [R, jnp.zeros((3,3))],
            [p_hat @ R, R]
        ]) 
        return mat
    
    def inertia_tensor(self, q):
        """
        Computes the mass matrix M(q) using RNEA.
        
        Args:
            q (jnp.ndarray): Joint positions (ndof,)

        Returns:
            jnp.ndarray: Mass matrix (ndof, ndof)
        """
        n = len(q)
        M = jnp.zeros((n, n))

        def column_rnea(i):
            e_i = jnp.zeros(n).at[i].set(1.0)   
            _, _, _, f = self._rnea(q, jnp.zeros(n), e_i)  
            return f  

        M = jax.vmap(column_rnea)(jnp.arange(n))  
        
        return M
      
    def coriolis_tensor(self, q, qp):
        """
        Computes the Coriolis matrix C(q, qp) using automatic differentiation in JAX.

        Args:
            q (jnp.ndarray): Joint positions (ndof,)
            qp (jnp.ndarray): Joint velocities (ndof,)

        Returns:
            jnp.ndarray: Coriolis matrix C(q, qp) of shape (ndof, ndof)
        """
        n = len(q)
        
        def rnea_coriolis(qp_var):
            c, _, _, _ = self._rnea(q, qp_var, jnp.zeros(n))  
            return c  
        C = jax.jacfwd(rnea_coriolis)(qp)

        return C
    
    def gravity_torques(self, q:jnp.ndarray=None):
        """ 
        Computes the joints gravity torques given a configuration vector.
        
        Args:
            - q :  joints position vector (nq * 1)
        Returns:
            - tau_g : numpy.ndarray.    
        """
        if q is None:
            q = self.q
        qp = jnp.zeros_like(q)
        _, _, _, f = self._rnea(q, qp, qpp=None)
        tau_g = jnp.zeros_like(q)
        for i in range(len(q)):
            S = self._screw(i)
            tau_g = tau_g.at[i].set(jnp.matmul(jnp.transpose(S), f[:, i])) 
            
        return tau_g
    
    def generalized_forces(self, q=None, qp=None, qpp=None)->jnp.ndarray:
        """
        Compute the genralized forces for each link using the 
        recursive netwon euler alogrithm.
         
        Args:
            - q    : Joints position vector. ( nq * 1 )
            - qp   : Joints velocity vector. ( nq * 1 )
            - qpp  : Joints acceleration vector. ( nq * 1 )
        Returns:
            - f  : (rotation + translation) compennat forces 
        """
        q = q if q is not None else self.q
        qp = qp if qp is not None else self.v
        qpp = qpp if qpp is not None else self.v
        f = jnp.zeros((6, self.ndof))
        _, _, _, f = self._rnea(q, qp, qpp)
      
        return f
    
    def generalized_torques(self, q=None, qp=None, qpp=None)->jnp.ndarray:
        """
        Return the genralized torques compennat using the recursive
        netwon euler alogrithm.
        """
        f = self.generalized_forces(q, qp, qpp)
        return f[-3:,:]
    
    def full_forces(self, alpha:jnp.array, beta:jnp.array, gamma:jnp.array,
        q=None, qp=None, qpp=None)->jnp.ndarray:
        """ 
        Compute the joint torques given the friction effects and damping 
        
        """
        
        return jnp.ndarray()
    
    def full_torques(self, alpha:jnp.array, beta:jnp.array, gamma:jnp.array,
        q=None, qp=None, qpp=None)->jnp.ndarray:
        """ 
        Compute the joint torques given the friction effects and damping (if not given)
        
        """
        
        return jnp.ndarray()