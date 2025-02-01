from setup_tests import *
from dynamapp.actuators import BLDC

class TestBLDC(unittest.TestCase):
    
    def test_torque_not_none(self):
        d2Q_d2t = np.linspace(0,10,100)
        dQ_dt = discreteTimeIntegral(d2Q_d2t,0.001)
        Q_t = discreteTimeIntegral(dQ_dt,0.001)
        
        motor = BLDC(Q_t,dQ_dt,d2Q_d2t)
        tau_m = motor.computeOutputTorque()
        self.assertIsNotNone(tau_m)
        
    def test_current_not_none(self):
        d2Q_d2t = np.linspace(0,10,100)
        dQ_dt = discreteTimeIntegral(d2Q_d2t,0.001)
        Q_t = discreteTimeIntegral(dQ_dt,0.001)
        
        motor = BLDC(Q_t,dQ_dt,d2Q_d2t)
        I = motor.computeArmatureCurrent()
        self.assertIsNotNone(I)

def plot_torque():
    d2Q_d2t = np.linspace(0,10,1000)
    dQ_dt = discreteTimeIntegral(d2Q_d2t,0.001)
    Q_t = discreteTimeIntegral(dQ_dt,0.001)    
    motor = BLDC(Q_t,dQ_dt,d2Q_d2t,0.7,1.91,0.14)
    tau_m = motor.computeOutputTorque()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(dQ_dt .size), y=dQ_dt , linewidth=0.5, color='green', label='velocity')
    sns.lineplot(x=np.arange(tau_m.size), y=tau_m, linewidth=0.5, color='blue', label='torque')
    sns.lineplot(x=np.arange(d2Q_d2t.size), y=d2Q_d2t, linewidth=0.5, color='red', label='acceleration')
    plt.show()
    
def plot_current():
    d2Q_d2t = np.linspace(0,10,1000)
    dQ_dt = discreteTimeIntegral(d2Q_d2t,0.001)
    Q_t = discreteTimeIntegral(dQ_dt,0.001)    
    motor = BLDC(Q_t,dQ_dt,d2Q_d2t,0.7,1.1,0.14)
    I = motor.computeArmatureCurrent()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(dQ_dt .size), y=dQ_dt , linewidth=0.5, color='green', label='velocity')
    sns.lineplot(x=np.arange(I.size), y=I, linewidth=0.5, color='blue', label='current')
    sns.lineplot(x=np.arange(d2Q_d2t.size), y=d2Q_d2t, linewidth=0.5, color='red', label='acceleration')
    plt.show()
     
if __name__ == "__main__":
    unittest.main(exit=False) 
    plot_torque()
    plot_current()
    
        