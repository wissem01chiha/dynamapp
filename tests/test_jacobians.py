from setup_tests import *
from dynamapp.jacobians import *

class TestJacobians(unittest.TestCase):
    
    def setUp(self):
    
        Imats = [jnp.full((6, 6), fill_value=5.0) for _ in range(3)]
        dhparams = [
            [1, 5.47, 0.5, jnp.pi / 3],  
            [0.5, 1, 0.47, jnp.pi / 6],
            [1, 4, 0.0, jnp.pi / 7]
        ]
        dampings = [1.0, 2, 6]
        x_init_ = jnp.zeros((6, 1)) 
        
        self.m = ModelJacobian(Imats, dhparams,dampings=dampings)
        self.ms = ModelStateJacobian(Imats,dhparams,-9.81,dampings=dampings,x_init=x_init_)
        self.q = jnp.array([0.1, 0.2, 0.3])
        self.v = jnp.array([0.4, 0.5, 0.6])
        self.a = jnp.array([0.7, 0.8, 0.9])
        self.tensor = jnp.ones((6, 6, self.m.ndof))  
    
    def test_generalized_torques_wrt_inertia(self):
         
        W = self.m.generalized_torques_wrt_inertia(self.q, self.v, self.a)
        self.assertEqual(W.size, (self.m.ndof * self.m.ndof * self.m.ndof * 36))  
        self.assertTrue(jnp.all(jnp.isfinite(W))) 
        
    def test_inertia_tensor_wrt_inertia(self):
        
        W = self.m.inertia_tensor_wrt_inertia(self.q)
        self.assertEqual(W.size, (self.m.ndof *self.m.ndof * self.m.ndof * 36 ))  
        self.assertTrue(jnp.all(jnp.isfinite(W))) 
        
    def test_generalized_torques_wrt_dhparms(self):
         
        W = self.m.generalized_torques_wrt_dhparams(self.q, self.v, self.a)
        self.assertEqual(W.size, (self.m.ndof * self.m.ndof * self.m.ndof * 4 ))  
        self.assertTrue(jnp.all(jnp.isfinite(W)))
        
    def test_generalized_torques_wrt_damping(self):
        
        W = self.m.generalized_torques_wrt_damping(self.q, self.v, self.a)
        self.assertEqual(W.size, (self.m.ndof * self.m.ndof * self.m.ndof  ))  
        self.assertTrue(jnp.all(jnp.isfinite(W)))

    def test_state_matrix_a_wrt_state(self):
        
        W = self.ms.state_matrix_a_wrt_state(self.q,self.v)
        self.assertEqual(W.size, (2* self.m.ndof * 2 * self.m.ndof *2* self.m.ndof)) 
        self.assertTrue(jnp.all(jnp.isfinite(W)))
        
if __name__ == "__main__":
    unittest.main()