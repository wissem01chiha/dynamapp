from setup_tests import *
from dynamapp_old.kalman import Kalman
from dynamapp_old.state_space import StateSpace

class TestKalman(unittest.TestCase):
    def setUp(self) -> None:
        self.x_init = jnp.array([[2]])
        self.model = StateSpace(
            jnp.array([[.5]]),
            jnp.array([[.6]]),
            jnp.array([[.7]]),
            jnp.array([[.8]]),
            jnp.array([[1]]),
            x_init=self.x_init
        )

    def test_step(self):
        n_datapoints = 100
        noise_reduction = 1e5

        kalman = Kalman(copy.deepcopy(self.model), jnp.eye(2) / (noise_reduction ** 2))
        jnp.random.seed(0)
        for _ in range(n_datapoints):
            u = jnp.random.standard_normal((1, 1))

            y = self.model.step(u, jnp.random.standard_normal((1, 1)) / noise_reduction)
            kalman.step(y, u)

        self.assertTrue(is_slightly_close(
            2.0593793476904603,
            kalman.y_filtereds[-1]
        ))

    def test_step_with_nans(self):
        n_datapoints = 5

        kalman = Kalman(copy.deepcopy(self.model), jnp.eye(2))
        jnp.random.seed(0)
        for i in range(n_datapoints):
            u = jnp.random.standard_normal((1, 1))

            y = self.model.step(u, jnp.random.standard_normal((1, 1)))
            if i == 3:
                y = None  
            kalman.step(y, u)

        self.assertEqual(
            [jnp.isnan(x[0, 0]) for x in kalman.ys],
            [False, False, False, True, False]
        )
        self.assertTrue(is_slightly_close(
            1.3813007321174262,
            kalman.ys[-1]
        ))

    def test_extrapolate(self):
        n_to_predict = 4
        kalman = Kalman(copy.deepcopy(self.model), jnp.eye(2))
        kalman.x_predicteds = [
            self.x_init
        ]
        predictions = kalman.extrapolate(n_to_predict)

        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([
                [1.4],
                [.7],
                [.35],
                [.175]
            ]),
            predictions.to_numpy()
        )))

    def test_measurement_and_state_standard_deviation(self):
        n_steps = 3
        noise_variance = 9
        state_variance = 16 / self.model.c[0, 0] ** 2

        kalman = Kalman(self.model, noise_variance * jnp.eye(2))
        state_covariance_matrices = [
            state_variance * jnp.eye(1) for _ in range(n_steps)
        ]

        output_standard_deviations = kalman._measurement_and_state_standard_deviation(state_covariance_matrices)

        self.assertEqual(n_steps, len(output_standard_deviations))
        for output_standard_deviation in output_standard_deviations:
            self.assertTrue(jnp.isclose(5, output_standard_deviation))

    def test_list_of_states_to_array(self):
        list_of_states = [
            jnp.array([[1], [2]]),
            jnp.array([[3], [4]])
        ]
        result = Kalman._list_of_states_to_array(list_of_states)
        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([
                [1, 2],
                [3, 4]
            ]),
            result
        )))

    def test_to_dataframe(self):
        kalman = Kalman(self.model, jnp.eye(2))
        kalman.us = [
            jnp.array([[1]]),
            jnp.array([[1]]),
        ]
        kalman.ys = [
            jnp.array([[1]]),
            jnp.array([[3]]),
        ]
        kalman.y_filtereds = [
            jnp.array([[11]]),
            jnp.array([[13]]),
        ]
        kalman.y_predicteds = [
            jnp.array([[21]]),
            jnp.array([[23]]),
        ]
        kalman.p_filtereds = 2 * [jnp.eye(1)]
        kalman.p_predicteds = 2 * [jnp.eye(1)]
        df = kalman.to_dataframe()

        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([1, 3]),
            df[('$y_0$', kalman.actual_label, kalman.output_label)].to_numpy()
        )))
        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([11, 13]),
            df[('$y_0$', kalman.filtered_label, kalman.output_label)].to_numpy()
        )))
        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([21, 23]),
            df[('$y_0$', kalman.next_predicted_label, kalman.output_label)].to_numpy()
        )))
        self.assertTrue(jnp.all(jnp.isclose(
            jnp.array([21.8]),
            df[('$y_0$', kalman.next_predicted_corrected_label, kalman.output_label)].to_numpy()[:-1]
        )))
        self.assertTrue(jnp.isnan(
            df[('$y_0$', kalman.next_predicted_corrected_label, kalman.output_label)].iloc[-1]
        ))

def is_slightly_close(matrix, number):
    return jnp.isclose(matrix, number, rtol=0, atol=1e-3)

if __name__ == "__main__":
    unittest.main() 