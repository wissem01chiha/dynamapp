from setup_tests import *
from dynamapp_old.generators import *

class TestModelDataGenerator(unittest.TestCase):

    def setUp(self):
        
        Imats = [jnp.full((6, 6), fill_value=5.0) for _ in range(3)]
        dhparams = [
            [1, 5.47, 0.5, jnp.pi / 3],  
            [0.5, 1, 0.47, jnp.pi / 6],
            [1, 4, 0.0, jnp.pi / 7]
        ]
        dampings = [1.0, 2, 6]

        self.model = Model(Imats, dhparams,dampings=dampings)  
        control_points = jnp.linspace(0, 10, 100)   
        self.trajectory = SplineTrajectory(ndof=3, sampling=100, ti=0.0, tf=10.0, control_points=control_points)
        self.data_generator = ModelDataGenerator(self.model, self.trajectory)


    def test_compute_velocities(self):
         
        q_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        q_dot_data = self.data_generator.compute_velocities(q_data)
        self.assertEqual(q_dot_data.shape,(2,3))
        self.assertIsInstance(q_dot_data, jnp.ndarray)
         

    def test_compute_accelerations(self):
         
        q_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        q_dot_data = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        q_ddot_data = self.data_generator.compute_accelerations(q_data, q_dot_data)
        self.assertIsInstance(q_ddot_data, jnp.ndarray)
        self.assertEqual(q_ddot_data.shape,(2,3))
        

    def test_compute_torques(self):
       
        q_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        q_dot_data = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        q_ddot_data = jnp.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        
        tau_data = self.data_generator.compute_torques(q_data, q_dot_data, q_ddot_data)
        self.assertIsInstance(tau_data, jnp.ndarray)
        self.assertEqual(tau_data.shape, (2,3))
        
    def test_generate_trajectory_data(self):
         
        data = self.data_generator.generate_trajectory_data()
        self.assertIsInstance(data, dict)
        self.assertIn('q', data)
        self.assertIn('q_dot', data)
        self.assertIn('q_ddot', data)
        self.assertIn('tau', data)
        
class TestModelStateDataGenerator(unittest.TestCase):

    def setUp(self):
        
        Imats = [jnp.full((6, 6), fill_value=5.0) for _ in range(3)]
        dhparams = [
            [1, 5.47, 0.5, jnp.pi / 3],  
            [0.5, 1, 0.47, jnp.pi / 6],
            [1, 4, 0.0, jnp.pi / 7]
        ]
        dampings = [1.0, 2, 6]

        self.x_init = jnp.zeros((6, 1))  
        self.model_state = ModelState(Imats, dhparams, gravity=-9.81,x_init=self.x_init)
        self.data_generator = ModelStateDataGenerator(self.model_state, num_samples=5, time_steps=10)

    def test_generate_data(self):
      
        x_data, u_data, y_data = self.data_generator.generate_data()
        self.assertIsInstance(x_data, jnp.ndarray)
        self.assertIsInstance(u_data, jnp.ndarray)
        self.assertIsInstance(y_data, jnp.ndarray)
 

    def test_compute_data_statistics(self):

        x_data = jnp.zeros((10, 100, 6))   
        u_data = jnp.zeros((10, 100, 3))   
        y_data = jnp.zeros((10, 100, 3))  
        stats = self.data_generator.compute_data_statistics(x_data, u_data, y_data)
        self.assertIsInstance(stats, dict)
        self.assertIn('x_mean', stats)
        self.assertIn('x_std', stats)
        self.assertIn('u_mean', stats)
        self.assertIn('u_std', stats)
        self.assertIn('y_mean', stats)
        self.assertIn('y_std', stats)

if __name__ == '__main__':
    unittest.main()