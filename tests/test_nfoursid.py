from setup_tests import *
from dynamapp.nfoursid import NFourSID
from dynamapp.state_space import StateSpace

class TestNFourSid(unittest.TestCase):
    
    def test_which_is_actually_regression_test(self):
        
        n_datapoints = 100
        model = StateSpace(
            jnp.array([[.5]]),
            jnp.array([[.6]]),
            jnp.array([[.7]]),
            jnp.array([[.8]]),
        )
        key =random.PRNGKey(0) 
        for _ in range(n_datapoints):
            model.step(random.normal(key, (1, 1)))

        nfoursid = NFourSID(
            model.to_dataframe(),
            model.y_column_names,
            input_columns=model.u_column_names,
            num_block_rows=2
        )
        nfoursid.subspace_identification()
        identified_model, covariance_matrix = nfoursid.system_identification(rank=1)

        self.assertTrue(is_slightly_close(.5, identified_model.a))
        self.assertTrue(is_slightly_close(.8, identified_model.d))
        self.assertTrue(jnp.all(is_slightly_close(0, covariance_matrix)))


def is_slightly_close(matrix, number):
    return jnp.isclose(matrix, number, rtol=0, atol=1e-3)

if __name__ == "__main__":
    unittest.main()