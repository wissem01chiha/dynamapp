from setup_tests import *
from dynamapp.regressors import *
from dynamapp.model import Model

class TestRegressor(unittest.TestCase):
    def setUp(self):
        
        Imats = [jnp.full((6, 6), fill_value=5.0) for _ in range(3)]
        dhparams = [
            [1, 5.47, 0.5, np.pi / 3],  
            [0.5, 1, 0.47, np.pi / 6],
            [1, 4, 0.0, np.pi / 7]
        ]
        dampings = [1.0, 2, 6]

        self.m = Model(Imats, dhparams,dampings=dampings)
        self.q = jnp.array([0.1, 0.2, 0.3])
        self.v = jnp.array([0.4, 0.5, 0.6])
        self.a = jnp.array([0.7, 0.8, 0.9])
        self.tensor = jnp.ones((6, 6, self.m.ndof))  
    
    def test_generalized_torques_wrt_inertia(self):
         
        W = generalized_torques_wrt_inertia(self.m, self.q, self.v, self.a)
        self.assertEqual(W.size, (self.m.ndof * self.m.ndof * self.m.ndof * 36 ))  
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

if __name__ == "__main__":
    unittest.main()