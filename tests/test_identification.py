from setup_tests import *
from dynamapp.identification import IDIMNLS
 
class TestIDIMNLS(unittest.TestCase):
    
    @staticmethod
    def modelFun(params):
        t = np.linspace(0, 2*np.pi, 1000)
        array = np.zeros((1000,7))
        for i in range(7):
            array[:, i] = np.sin(params[0] * t + params[1] * i) \
                * params[2] + params[3] * np.cos(params[4] * t + params[5] * i)
        return array
    
    def test_cost_function_not_none(self):
        output = np.random.rand(1000,7)
        model = IDIMNLS(6,output,TestIDIMNLS.modelFun)
        obj = model.computeLsCostFunction(np.random.rand(6))
        self.assertIsNotNone(obj)
        
    def test_cost_function_shape(self):
        output = np.random.rand(1000,7)
        model = IDIMNLS(6,output,TestIDIMNLS.modelFun)
        obj = model.computeLsCostFunction(np.random.rand(6))
        self.assertEqual(obj.ndim==0,True)
        
    def test_optimize_not_none(self):
        output = 1.2*np.random.normal(0,0.5,size=(1000,7))
        model = IDIMNLS(6,output,TestIDIMNLS.modelFun)
        sol = model.optimize(np.zeros(6))
        self.assertIsNotNone(sol)
        
    def test_evaluate(self):
        output = self.modelFun([0.5,1,0.3,0.01,0.5,1.5])
        model = IDIMNLS(6,output,TestIDIMNLS.modelFun)
        sol = model.optimize(np.random.rand(6))
        self.assertIsNotNone(model.optimized_params)
        model.visualizeResults('results')
        
    def test_plot_cost_function(self):
        output = self.modelFun([0.8,1.1,0.7,0.1,0.85,1.5])
        model = IDIMNLS(6,output,TestIDIMNLS.modelFun)
        model.optimize(np.random.rand(6))
        print(model.optimized_params)
        model.visualizeCostFunction()
        
        
if __name__ == "__main__":
    unittest.main(exit=False)
    plt.show() 