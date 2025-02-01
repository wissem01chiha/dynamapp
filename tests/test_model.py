from setup_tests import *
from dynamapp.model import Model   

class TestModel(unittest.TestCase):
    
    def setUp(self) -> None:

        Imats = [jnp.full((6, 6), fill_value=5.0) for _ in range(7)]
        Imats[0] = Imats[0].at[0, 0].set(10.0)   
        Imats[1] = Imats[1].at[1, 2].set(-5.0)   
        Imats[2] = Imats[2].at[4, 5].set(7.3)
        Imats[2] = Imats[2].at[4, 5].set(7.3)
        Imats[4] = Imats[5].at[2, 5].set(88.3)
        Imats[4] = Imats[5].at[5, 5].set(-0.3)
        Imats[4] = Imats[6].at[3, 1].set(-8.3)
        dhparams = [
            [1, 5.47, 0.5, np.pi / 3],  
            [0.5, 1, 0.47, np.pi / 6],
            [1, 4, 0.0, np.pi / 7],
            [-2, 5, 0.28, np.pi / 4],
            [-2, 1, 0.89, np.pi / 6],
            [1, 1, 0.8, np.pi / 3],
            [-3, -0.02, 0.5, np.pi / 4]
        ]
        self.model_ = Model(Imats, dhparams)

    def test_inertia_tensor_is_not_none(self):
        q = jnp.array([1.0, -8.5, -0.3, 87.0, 8.2, -10.4, 8.8])
        M = self.model_.inertia_tensor(q) 
        print(M)
        print(jnp.shape(M))
        self.assertIsNotNone(M, "The computed inertia matrix is None")
        
    def test_coriolis_matrix_not_none(self):
        q = jnp.zeros(7)  
        qp = jnp.zeros(7)  
        C = self.model_.coriolis_tensor(q, qp)
        self.assertIsNotNone(C, "The computed Coriolis matrix is None")
        
    def test_gravity_torque_not_none(self):
        q = jnp.zeros(7)  
        tau_g = self.model_.gravity_torques(q)
        self.assertIsNotNone(tau_g, "Gravity torques should not be None")
        
    def test_gravity_torque_shape(self):
        q = jnp.ones(7)  
        tau_g = self.model_.gravity_torques(q)
        self.assertEqual(tau_g.shape, q.shape, "Gravity torques shape mismatch")
        
    def test_gravity_torque_with_default_q(self):
        tau_g = self.model_.gravity_torques()
        self.assertIsNotNone(tau_g, "Gravity torques should not be None")
        
    def test_generalized_torques(self):
        q = np.ones(7)
        qp = np.ones(7)
        qpp = np.ones(7)
        torques = self.model_.generalized_torques(q, qp, qpp)
        self.assertIsNotNone(torques, "Generalized torques should not be None")

if __name__ == "__main__":
    unittest.main()
