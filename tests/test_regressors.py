from setup_tests import *
from dynamapp.regressors import *
from dynamapp.model import Model
from dynamapp.model_state import ModelState

class TestRegressor(unittest.TestCase):
    
    def setUp(self):
    
        Imats = [jnp.full((6, 6), fill_value=5.0) for _ in range(3)]
        dhparams = [
            [1, 5.47, 0.5, jnp.pi / 3],  
            [0.5, 1, 0.47, jnp.pi / 6],
            [1, 4, 0.0, jnp.pi / 7]
        ]
        dampings = [1.0, 2, 6]
        x_init_ = jnp.zeros((6, 1)) 
        
        self.m = Model(Imats, dhparams,dampings=dampings)
        self.ms = ModelState(Imats,dhparams,-9.81,dampings=dampings,x_init=x_init_)
        self.q = jnp.array([0.1, 0.2, 0.3])
        self.v = jnp.array([0.4, 0.5, 0.6])
        self.a = jnp.array([0.7, 0.8, 0.9])
        self.tensor = jnp.ones((6, 6, self.m.ndof))  
    
    def test_generalized_torques_wrt_inertia(self):
         
        W = generalized_torques_wrt_inertia(self.m, self.q, self.v, self.a)
        self.assertEqual(W.size, (self.m.ndof * self.m.ndof * self.m.ndof * 36))  
        self.assertTrue(jnp.all(jnp.isfinite(W))) 
        
    def test_inertia_tensor_wrt_inertia(self):
        
        W = inertia_tensor_wrt_inertia(self.m,self.q)
        self.assertEqual(W.size, (self.m.ndof *self.m.ndof * self.m.ndof * 36 ))  
        self.assertTrue(jnp.all(jnp.isfinite(W))) 
        
    def test_generalized_torques_wrt_dhparms(self):
         
        W = generalized_torques_wrt_dhparams(self.m, self.q, self.v, self.a)
        self.assertEqual(W.size, (self.m.ndof * self.m.ndof * self.m.ndof * 4 ))  
        self.assertTrue(jnp.all(jnp.isfinite(W)))
        
    def test_generalized_torques_wrt_damping(self):
        
        W = generalized_torques_wrt_damping(self.m, self.q, self.v, self.a)
        self.assertEqual(W.size, (self.m.ndof * self.m.ndof * self.m.ndof  ))  
        self.assertTrue(jnp.all(jnp.isfinite(W)))

    def test_state_matrix_a_wrt_state(self):
        
        W = state_matrix_a_wrt_state(self.ms,self.q,self.v)
        self.assertEqual(W.size, (2* self.m.ndof * 2 * self.m.ndof *2* self.m.ndof)) 
        self.assertTrue(jnp.all(jnp.isfinite(W)))
        
if __name__ == "__main__":
    unittest.main()