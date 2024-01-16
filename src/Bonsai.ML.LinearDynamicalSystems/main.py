from lds.inference import OnlineKalmanFilter
import numpy as np

DEFAULT_PARAMS = {
    "pos_x0" : 0, 
    "pos_y0" : 0,
    "vel_x0" : 0, 
    "vel_y0" : 0,
    "acc_x0" : 0, 
    "acc_y0" : 0,
    "sigma_a" : 10000, 
    "sigma_x" : 100,
    "sigma_y" : 100,
    "sqrt_diag_V0_value" : 0.001,
    "fps" : 60
}

class Model(OnlineKalmanFilter):

    def __init__(self,
                    pos_x0: float,
                    pos_y0: float,
                    vel_x0: float,
                    vel_y0: float,
                    acc_x0: float,
                    acc_y0: float,
                    sigma_a: float,
                    sigma_x: float,
                    sigma_y: float,
                    sqrt_diag_V0_value: float,
                    fps: int
                    ) -> None:
        
        super(OnlineKalmanFilter, self).__init__()

        self.pos_x0=pos_x0
        self.pos_y0=pos_y0
        self.vel_x0=vel_x0
        self.vel_y0=vel_y0
        self.acc_x0=acc_x0
        self.acc_y0=acc_y0
        self.sigma_a=sigma_a
        self.sigma_x=sigma_x
        self.sigma_y=sigma_y
        self.sqrt_diag_V0_value=sqrt_diag_V0_value
        self.fps=fps

        if np.isnan(self.pos_x0):
            self.pos_x0 = 0

        if np.isnan(self.pos_y0):
            self.pos_y0 = 0

        # build KF matrices for tracking
        dt = 1.0 / self.fps
        # Taken from the book
        # barShalomEtAl01-estimationWithApplicationToTrackingAndNavigation.pdf
        # section 6.3.3
            # Eq. 6.3.3-2
        B = np.array([  [1,     dt,     .5*dt**2,   0,      0,      0],
                        [0,     1,      dt,         0,      0,      0],
                        [0,     0,      1,          0,      0,      0],
                        [0,     0,      0,          1,      dt,     0.5*dt**2],
                        [0,     0,      0,          0,      1,      dt],
                        [0,     0,      0,          0,      0,      1]],
                      dtype=np.double)
        Z = np.array([  [1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0]],
                      dtype=np.double)
        Qt = np.array([ [dt**4/4,   dt**3/2,    dt**2/2,    0,          0,          0],
                        [dt**3/2,   dt**2,      dt,         0,          0,          0],
                        [dt**2/2,   dt,         1,          0,          0,          0],
                        [0,         0,          0,          dt**4/4,    dt**3/2,    dt**2/2],
                        [0,         0,          0,          dt**3/2,    dt**2,      dt],
                        [0,         0,          0,          dt**2/2,    dt,         1]],
                        dtype=np.double)

        R = np.diag([self.sigma_x**2, self.sigma_y**2])
        m0 = np.array([[self.pos_x0, self.vel_x0, self.acc_x0, self.pos_y0, self.vel_y0, self.acc_y0]], dtype=np.double).T
        V0 = np.diag(np.ones(len(m0))*self.sqrt_diag_V0_value**2)
        Q = Qt * self.sigma_a

        super().__init__(B, Q, m0, V0, Z, R)

    def update(self, x, y):

        if x is None:
            x = np.nan
        if y is None:
            y = np.nan

        return super().update(y=np.array([x, y]))
    
    def calculate_ellipse(self):
        cov_pos = self.P[0:6:3,0:6:3]
        self.el_w = 2 * np.sqrt(cov_pos[0,0])
        self.el_h = 2 * np.sqrt(cov_pos[1,1])
        self.el_a = 0.5 * np.arctan2(2 * cov_pos[0,1], cov_pos[0,0] - cov_pos[1,1])
    
if __name__ == "__main__":

    model = Model(**DEFAULT_PARAMS)
    x, y = 100, 100
    model.predict()
    model.update(x, y)
    print(model.x)
    print(model.x.shape)
    print(model.P.shape)