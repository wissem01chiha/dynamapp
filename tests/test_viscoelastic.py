from setup_tests import *
from dynamapp_old.viscoelastic import *

class TestFrictionModels(unittest.TestCase):

    def test_compute_coulomb_friction_force(self):
        v = jnp.array([1.0, -1.0, 0.5, -0.5])  
        fc = jnp.array([0.5, 0.5, 0.5, 0.5])   
        fs = jnp.array([0.1, 0.1, 0.1, 0.1])  
        
        expected_output = fc * jnp.sign(v) + fs * v

        result = compute_coulomb_friction_force(v, fc, fs)
        self.assertAlmostEqual(result.all(), expected_output.all(), 1e-6)

    def test_compute_friction_force(self):
         
        alpha = jnp.array([1.0, 2.0, 3.0])  
        beta = jnp.array([0.5, -0.3])  
        gamma = jnp.array([0.1, 0.2])   
        q = 1.0   
        v = 0.5   
        a = -0.2   
        expected_output = jnp.polyval(alpha, q) + jnp.polyval(beta, v) + jnp.polyval(gamma, a)
        result = compute_friction_force(alpha, beta, gamma, q, v, a)
        self.assertAlmostEqual(result.all(), expected_output.all(), 1e-6)


if __name__ == '__main__':
    unittest.main()