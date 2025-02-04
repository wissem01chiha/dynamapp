from setup_tests import *
from dynamapp.math_utils  import *

class TestUtils(unittest.TestCase):
    
    def test_block_hankel_matrix(self):
        
        matrix = jnp.array(range(15)).reshape((5, 3))
        hankel = block_hankel_matrix(matrix, 2)
        desired_result = jnp.array([
            [0., 3., 6., 9.],
            [1., 4., 7., 10.],
            [2., 5., 8., 11.],
            [3., 6., 9., 12.],
            [4., 7., 10., 13.],
            [5., 8., 11., 14.],
        ])
        self.assertTrue(jnp.all(jnp.isclose(desired_result, hankel)))

    def test_eigenvalue_decomposition(self):
        
        matrix = jnp.fliplr(jnp.diag(jnp.array(range(1, 3))))
        decomposition = eigenvalue_decomposition(matrix)
        self.assertTrue(jnp.all(jnp.isclose(
            [[0, -1],
             [-1, 0]],
            decomposition.left_orthogonal
        )))
        self.assertTrue(jnp.all(jnp.isclose(
            [2, 1],
            jnp.diagonal(decomposition.eigenvalues)
        )))
        self.assertTrue(jnp.all(jnp.isclose(
            [[-1, 0],
             [0, -1]],
            decomposition.right_orthogonal
        )))

        reduced_decomposition = reduce_decomposition(decomposition, 1)
        self.assertTrue(jnp.all(jnp.isclose(
            [[0], [-1]],
            reduced_decomposition.left_orthogonal
        )))
        self.assertTrue(jnp.all(jnp.isclose(
            [[2]],
            reduced_decomposition.eigenvalues
        )))
        self.assertTrue(jnp.all(jnp.isclose(
            [[-1, 0]],
            reduced_decomposition.right_orthogonal
        )))

    def test_vectorize(self):
        
        matrix = jnp.array([
            [0, 2],
            [1, 3]
        ])
        result = vectorize(matrix)
        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([
                [0],
                [1],
                [2],
                [3],
            ]),
            result
        )))

    def test_unvectorize(self):
        
        matrix = jnp.array([
            [0],
            [1],
            [2],
            [3],
        ])
        result = unvectorize(matrix, num_rows=2)
        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([
                [0, 2],
                [1, 3]
            ]),
            result
        )))

        with self.assertRaises(ValueError):
            unvectorize(matrix, num_rows=3)

        incompatible_matrix = matrix.T
        with self.assertRaises(ValueError):
            unvectorize(incompatible_matrix, 1)

    def test_validate_matrix_shape(self):
        
        with self.assertRaises(ValueError):
            validate_matrix_shape(
                jnp.array([[0]]),
                (42),
                'error'
            )
            
if __name__ == "__main__":
    unittest.main() 