
from setup_tests import *
from dynamapp.regressors import Regressor

class TestRegressor(unittest.TestCase):
    
    def test_full_regressor_not_none(self):
        reg = Regressor()
        q = np.random.rand(100,7)
        v = np.random.rand(100,7)
        a = np.random.rand(100,7)
        W = reg.computeFullRegressor(q,v,a)
        self.assertIsNotNone(W)
        
    def test_basic_regressor_not_none(self):
        reg = Regressor()
        q = np.random.rand(7)
        v = np.random.rand(7)
        a = np.random.rand(7)
        W = reg.computeBasicRegressor(q,v,a)
        self.assertIsNotNone(W)
        
    def test_reduced_regressor(self):
        reg = Regressor()
        q = np.random.rand(100,7)
        v = np.random.rand(100,7)
        a = np.random.rand(100,7)
        Wred = reg.computeReducedRegressor(q,v,a,1e-6)
        self.assertIsNotNone(Wred)
        
        
        
        
        
        
        
        
if __name__ == "__main__":
    unittest.main() 