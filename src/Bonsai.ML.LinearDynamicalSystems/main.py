from lds.inference import OnlineKalmanFilter, TimeVaryingOnlineKalmanFilter
import numpy as np
from scipy.stats import multivariate_normal

class KalmanFilterKinematics(OnlineKalmanFilter):

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

        dt = 1.0 / self.fps

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
    
class KalmanFilterLinearRegression(TimeVaryingOnlineKalmanFilter):

    def __init__(self,
                    likelihood_precision_coef: float,
                    prior_precision_coef: float,
                    n_features: int,
                    x: list[list[float]] = None,
                    P: list[list[float]] = None,
                    ) -> None:

        self.likelihood_precision_coef = likelihood_precision_coef
        self.prior_precision_coef = prior_precision_coef
        self.n_features = n_features

        if x is None:
            x = np.zeros((self.n_features,1), dtype=np.double)
        else:
            x = np.array(x)
        
        if P is None:
            P = 1.0 / self.prior_precision_coef * np.eye(len(x))
        else:
            P = np.array(P)

        self.x = x
        self.P = P

        self.B = np.eye(N=len(self.x))
        self.Q = np.zeros(shape=((len(self.x), len(self.x))))
        self.R = np.array([[1.0/self.likelihood_precision_coef]])

        super().__init__()

    def predict(self):
        self.x, self.P = super().predict(x = self.x, P = self.P, B = self.B, Q = self.Q)

    def update(self, x, y):
        if not isinstance(x, list):
            x = [x]
        self.x, self.P = super().update(y = np.array(y, dtype=np.float64), x = self.x, P = self.P, Z = np.array(x).reshape((1, -1)), R = self.R)
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]
    
    def pdf(self, x0 = 0, x1 = 1, xsteps = 100, y0 = 0, y1 = 1, ysteps = 100):

        self.x0 = x0
        self.x1 = x1
        self.xsteps = xsteps
        self.y0 = y0
        self.y1 = y1
        self.ysteps = ysteps
        
        rv = multivariate_normal(self.x.flatten(), self.P)
        xpos = np.linspace(x0, x1, xsteps)
        ypos = np.linspace(y0, y1, ysteps)
        xx, yy = np.meshgrid(xpos, ypos)
        pos = np.dstack((xx, yy))
        self.pdf_values = rv.pdf(pos)