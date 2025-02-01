from setup_tests import *
from dynamapp.viscoelastic import Dahl

class TestDahl(unittest.TestCase):
    def test_force_not_none(self):
        model = Dahl(0.1,1)
        F= model.computeFrictionForce(np.random.rand(1001))
        self.assertIsNotNone(F)
        
    def test_force_size(self):
        model = Dahl(1.2,1)
        velocity = np.random.rand(1001)
        F= model.computeFrictionForce(velocity)
        self.assertEqual(np.size(F),np.size(velocity))
    
    def test_force_dim(self):
        model =Dahl(1.2,1)
        velocity = np.random.rand(1001)
        F= model.computeFrictionForce(velocity)
        self.assertEqual(np.ndim(F),np.ndim(velocity))
        
    def test_froce_not_null(self):
        model = Dahl(100,15)
        velocity =  np.linspace(0,10,100)
        F = model.computeFrictionForce(velocity)
        self.assertNotEqual(np.all(F==0),True)
        
def plot_force(sigma0,Fs):
    model = Dahl(sigma0, Fs)
    velocity = np.linspace(0, 10, 100)
    F = model.computeFrictionForce(velocity)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(velocity.size), y=velocity, linewidth=0.5, color='blue', label='Velocity')
    sns.lineplot(x=np.arange(velocity.size), y=F, linewidth=0.5, color='red', label='Friction Force')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Velocity and Friction Force')
    plt.legend()
    plt.show()
        
from dynamapp.viscoelastic import MaxwellSlip

class TestRobot(unittest.TestCase):
    def test_friction_force_not_null(self):
        model = MaxwellSlip(3,np.ones(10),[1,2,3],[0.1,0.2,0.3],1)
        F = model.computeFrictionForce()
        self.assertIsNotNone(F)
        
    def test_friction_force_shape(self):
        v= np.ones(10)
        model = MaxwellSlip(3,v,[1,2,3],[0.1,0.2,0.3],1)
        F = model.computeFrictionForce()
        self.assertEqual(F.shape,v.shape)
        
    
if __name__ == "__main__":
    unittest.main(exit=False)     
    plot_force(100,15)  