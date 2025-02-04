from setup_tests import *
from dynamapp.model_state import ModelState
from dynamapp.model import Model

class TestModelState(unittest.TestCase):
    
    def setUp(self) -> None:
        
        Imats = [jnp.full((6, 6), fill_value=5.0) for _ in range(3)]
        dhparams = [
            [1, 5.47, 0.5, jnp.pi / 3],  
            [0.5, 1, 0.47, jnp.pi / 6],
            [1, 4, 0.0, jnp.pi / 7]
        ]
        dampings = [1.0, 2, 6]

        self.m = Model(Imats, dhparams,dampings=dampings)
        self.x_init = jnp.zeros((6, 1))  
        self.model_state = ModelState(Imats, dhparams, gravity=-9.81,x_init=self.x_init)

    def test_compute_matrices(self):
        
        x = jnp.zeros((6, 1))
        self.model_state._compute_matrices(x)
        self.assertEqual(self.model_state.model_state_space.a.shape, (6, 6))

    def test_output(self):
        
        x = jnp.zeros((6, 1))
        u = jnp.zeros((3, 1))
        e = jnp.zeros((3, 1))
        y = self.model_state.output(x, u, e)
        self.assertEqual(y.shape, (3, 1))

    def test_step(self):
        
        u = jnp.zeros((3, 1))
        e = jnp.zeros((3, 1))
        y = self.model_state.step(u, e)
        self.assertEqual(y.shape, (3, 1))

    def test_set_x_init(self):
        
        x_init = jnp.ones((6, 1))
        self.model_state.set_x_init(x_init)
        self.assertTrue(jnp.array_equal(self.model_state.x_init, x_init))

    def test_compute_eigvals(self):
        x = jnp.zeros((6, 1))
        eigvals = self.model_state.compute_eigvals(x)
        self.assertEqual(eigvals.shape[0], 6)

    def test_is_stable(self):
        
        x = jnp.zeros((6, 1))
        stability = self.model_state._is_stable(x)
        self.assertIsInstance(stability, bool)

    def test_lsim(self):
        
        u = jnp.zeros((3, 10))
        e = jnp.zeros((3, 10))
        xs = self.model_state.lsim(u, e)
        self.assertEqual(len(xs), 10)

    def test_compute_obs_matrix(self):
        
        x = jnp.zeros((6, 1))
        obs_matrix = self.model_state.compute_obs_matrix(x)
        self.assertEqual(obs_matrix.shape[0], 18)

    def test_compute_ctlb_matrix(self):
        
        x = jnp.zeros((6, 1))
        ctlb_matrix = self.model_state.compute_ctlb_matrix(x)
        self.assertEqual(ctlb_matrix.shape[1], 18)

    def test_get_state_matrix_a(self):
        
        x = jnp.zeros((6, 1))
        A = self.model_state.get_state_matrix_a(x)
        self.assertEqual(A.shape, (6, 6))

if __name__ == "__main__":
    unittest.main() 
        