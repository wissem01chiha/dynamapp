from setup_tests import *
from dynamapp.solvers import *

class TestSolvers(unittest.TestCase):
    
    def setUp(self) -> None:
        pass
    
class TestSolvers(unittest.TestCase):
    
    def setUp(self) -> None:
        pass
    
    def test_solve_least_square(self):
        W = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = jnp.array([[7.0], [8.0], [9.0]])
        X = solve_least_square(W, Y)
        self.assertEqual(X.shape, (2, 1))
        W_dot_X = jnp.dot(W, X)
        self.assertTrue(jnp.allclose(W_dot_X, Y, atol=1e-6))
    
    def test_solve_riccati_equation(self):
        A = jnp.array([[1.0, 0.5], [0.0, 1.0]])
        B = jnp.array([[0.5], [1.0]])
        Q = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        R = jnp.array([[1.0]])
        P = solve_riccati_equation(A, B, Q, R)
        self.assertEqual(P.shape, (2, 2))
        self.assertTrue(jnp.allclose(P, P.T, atol=1e-6))
        
    def test_luenberger_observer(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[0.5], [1.0]])
        C = jnp.array([[1.0, 0.0]])
        desired_poles = [-1.0, -2.0]
        L = luenberger_observer(A, B, C, desired_poles)
        self.assertEqual(L.shape, (2, 2))
        self.assertTrue(jnp.allclose(L, jnp.zeros_like(L)))

if __name__ == '__main__':
    unittest.main()