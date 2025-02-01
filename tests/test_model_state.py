from setup_tests import *
from dynamapp.model_state import StateSpace

class TestStateSpace(unittest.TestCase):
    
    def setUp(self) -> None:
        

        self.config_file_path = os.path.join(pkg_dir,"robot/kinova/config.yml")
        self.model = StateSpace(self.urdf_file_path,self.config_file_path)
    
    def test_state_matrices_not_none(self):
        A, B, C, D = self.model.computeStateMatrices(np.random.rand(14))
        self.assertIsNotNone(A)
        self.assertIsNotNone(B)
        self.assertIsNotNone(C)
        
    def test_state_matrices_shape(self):
        A, B, C, D = self.model.computeStateMatrices(np.random.rand(14))
        self.assertEqual(A.shape, (14,14))
        self.assertEqual(B.shape, (14,7))
        self.assertEqual(C.shape, (7,14))
        self.assertEqual(D.shape, (7,7))
        
    def test_state_input_vector_not_none(self):
        u = self.model.computeStateInputVector(states=np.random.rand(14,100),\
            input_torque=np.random.rand(100,7))
        self.assertIsNotNone(u)
       
    def test_state_input_vector_shape(self):
            u = self.model.computeStateInputVector(states=np.random.rand(14,100),\
            input_torque=np.random.rand(100,7))
            self.assertEqual(u.shape, (100,7))
         
    def test_update_state_vector(self):
        x_k_1 = self.model.updateStateVector(np.random.rand(14),np.random.rand(7))
        self.assertIsNotNone(x_k_1)
        self.assertEqual(x_k_1.size,14)
        self.assertEqual(np.all(x_k_1.shape ==(14,)),True)
           
    def test_simulate(self):
        states = self.model.lsim(x0=np.ones((14,)),input=np.ones((10,7)))
        self.assertIsNotNone(states)
        self.assertEqual(states.shape,(14,10), True)
        
    def test_augmented_state_not_none(self):
        x = np.random.rand(14,)
        A_aug, B_aug, C_aug, D_aug =self.model.computeAugmentedStateMatrices(x)
        self.assertIsNotNone(A_aug)
        self.assertIsNotNone(B_aug)
        self.assertIsNotNone(C_aug)
        self.assertIsNotNone(D_aug)
        
    def test_state_place_poles_not_none(self):
        AA = self.model.state_place_poles(-np.abs(np.random.rand(14,10)),\
            np.random.rand(14,10))
        self.assertIsNotNone(AA)
    
if __name__ == "__main__":
    unittest.main() 
        