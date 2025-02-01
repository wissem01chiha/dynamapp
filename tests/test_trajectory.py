from setup_tests import *
from dynamapp.trajectory import FourierGenerator

class TestFourier(unittest.TestCase):
    
    def setUp(self) -> None:
        np.random.seed()
        self.trajectory_parameters = {
        'samples': 120,
        'nbfourierterms': 5,
        'frequancy': 10,
        'ndof': 7,
        'Aij': [[1.2 for _ in range(5)] for _ in range(7)],  
        'Bij': [[2 for _ in range(5)] for _ in range(7)]
        }
        
    def test_compute_traj_state_not_none(self):
        traj = FourierGenerator(self.trajectory_parameters)
        q0 = np.zeros(7)
        qp0 = np.zeros(7)
        qpp0 = np.zeros(7)
        state =traj.computeTrajectoryState(1.2,q0,qp0,qpp0)
        self.assertIsNotNone(state)
        
    def test_traj_full_state_not_none(self):
        traj = FourierGenerator(self.trajectory_parameters)
        q0 = np.zeros(7)
        qp0 = np.zeros(7)
        qpp0 = np.zeros(7)
        Q, Qp, Qpp = traj.computeFullTrajectory(0,1,q0,qp0,qpp0)
        self.assertIsNotNone(Q)
        self.assertIsNotNone(Qp)
        self.assertIsNotNone(Qpp)
        
    def test_traj_full_state_shapes(self):
        traj = FourierGenerator(self.trajectory_parameters)
        q0 = np.zeros(7)
        qp0 = np.zeros(7)
        qpp0 = np.zeros(7)
        Q, Qp, Qpp = traj.computeFullTrajectory(0,1,q0,qp0,qpp0)
        self.assertEqual(Q.shape,(120,7))
        self.assertEqual(Qp.shape,(120,7))
        self.assertEqual(Qpp.shape,(120,7))
        
    def test_traj_criteria_not_none(self):
        self.trajectory_parameters['frequancy'] = 0.9
        traj =  FourierGenerator(self.trajectory_parameters)
        q0 = np.zeros(7)
        qp0 = np.zeros(7)
        qpp0 = np.zeros(7)
        x = np.ones(209)
        
        C = traj.computeTrajectoryCriterion(0,1,x,q0,qp0,qpp0)
        self.assertIsNotNone(C)
         
    """def test_traj_identificatbility(self):
        traj = FourierGenerator(self.trajectory_parameters)
        torque = np.random.rand(self.trajectory_parameters['samples'],7)
        q0 = np.zeros(7)
        qp0 = np.zeros(7)
        qpp0 =np.zeros(7)
        x = np.abs(np.random.rand(98))
        Q, Qp, Qpp = traj.computeFullTrajectory(0,1,q0,qp0,qpp0)
        traj.visualizeTrajectoryIdentifiability(0,1, torque, Q, Qp, Qpp, x)
        plt.show()
    """
        
if __name__ == "__main__":
    unittest.main() 