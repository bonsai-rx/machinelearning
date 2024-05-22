from lds.inference import OnlineKalmanFilter, TimeVaryingOnlineKalmanFilter
import numpy as np

import lds.learning
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

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
        self.Qe = np.array([ [dt**4/4,   dt**3/2,    dt**2/2,    0,          0,          0],
                        [dt**3/2,   dt**2,      dt,         0,          0,          0],
                        [dt**2/2,   dt,         1,          0,          0,          0],
                        [0,         0,          0,          dt**4/4,    dt**3/2,    dt**2/2],
                        [0,         0,          0,          dt**3/2,    dt**2,      dt],
                        [0,         0,          0,          dt**2/2,    dt,         1]],
                        dtype=np.double)

        R = np.diag([self.sigma_x**2, self.sigma_y**2]).astype(np.double)
        m0 = np.array([[self.pos_x0, self.vel_x0, self.acc_x0, self.pos_y0, self.vel_y0, self.acc_y0]], dtype=np.double).T
        V0 = np.diag(np.ones(len(m0))*self.sqrt_diag_V0_value**2).astype(np.double)
        Q = self.Qe * self.sigma_a

        self.batch = None
        self.is_running = False
        self._optimization_finished = False
        self.loop = None
        self.thread = None

        super().__init__(B, Q, m0, V0, Z, R)

    def update(self, x, y):

        if x is None:
            x = np.nan
        if y is None:
            y = np.nan

        return super().update(y=np.array([x, y]))
    
    def optimize(self, vars_to_estimate, max_iter, disp):
        sqrt_diag_R = np.array([self.sigma_x, self.sigma_y])
        m0 = self.m0.squeeze().copy()
        sqrt_diag_V0 = (np.ones(len(self.m0))*self.sqrt_diag_V0_value)

        y = self.batch.T.astype(np.double).copy()
        B = self.B.astype(np.double).copy()
        Qe = self.Qe.astype(np.double).copy()
        Z = self.Z.astype(np.double).copy()

        sigma_ax0 = sigma_ay0 = np.sqrt(self.sigma_a)

        optim_res_ga = lds.learning.scipy_optimize_SS_tracking_diagV0(y=y, B=B, sigma_ax0=sigma_ax0, sigma_ay0=sigma_ay0, Qe=Qe, Z=Z, sqrt_diag_R_0=sqrt_diag_R, m0_0=m0, sqrt_diag_V0_0=sqrt_diag_V0, max_iter=max_iter, disp=disp)

        if vars_to_estimate["sigma_a"]:
            self.sigma_a = optim_res_ga["x"]["sigma_ax"].item() ** 2
            self.Q = self.Qe*self.sigma_a

        if vars_to_estimate["m0"]:
            self.m0 = optim_res_ga["x"]["m0"][:, np.newaxis]

        if vars_to_estimate["V0"]:
            self.sqrt_diag_V0_value = optim_res_ga["x"]["sqrt_diag_V0"][0]
            self.V0 = np.diag(np.ones(len(self.m0))*self.sqrt_diag_V0_value**2).astype(np.double)
        
        if vars_to_estimate["R"]:
            self.sigma_x = optim_res_ga["x"]["sqrt_diag_R"][0]
            self.sigma_y = optim_res_ga["x"]["sqrt_diag_R"][1]
            self.R = np.diag([self.sigma_x**2, self.sigma_y**2]).astype(np.double)

    def run_optimization(self, 
                            x, 
                            y,
                            vars_to_estimate = None, 
                            batch_size = 20,
                            max_iter = 50,
                            disp = True):

        if self.batch is None:
            self.batch = np.array([[x, y]])

        elif len(self.batch) < batch_size:
            self.batch = np.vstack([self.batch, np.array([x, y])])

        if len(self.batch) == batch_size:

            if vars_to_estimate is None:
                vars_to_estimate = { 
                    "sigma_a": True, 
                    "sqrt_diag_R": True, 
                    "R" : True, 
                    "m0" : True, 
                    "sqrt_diag_V0": True, 
                    "V0" : True 
                }

            self.optimize(vars_to_estimate, max_iter, disp)
            self.batch = None

            return True
        
        return False
    
    def run_optimization_async(self, 
                                x, 
                                y,
                                vars_to_estimate = None, 
                                batch_size = 20,
                                max_iter = 50,
                                disp = True):

        if not self.is_running:

            if self.batch is None:
                self.batch = np.array([[x, y]])

            elif len(self.batch) < batch_size:
                self.batch = np.vstack([self.batch, np.array([x, y])])

            if len(self.batch) == batch_size:

                def start_loop(loop):
                    asyncio.set_event_loop(loop)
                    loop.run_forever()

                def on_completion(future):
                    self.batch = None
                    self.is_running = False
                    self._optimization_finished = True

                self.is_running = True

                if vars_to_estimate is None:
                    vars_to_estimate = { 
                        "sigma_a": True, 
                        "sqrt_diag_R": True, 
                        "R" : True, 
                        "m0" : True, 
                        "sqrt_diag_V0": True, 
                        "V0" : True 
                    }

                if self.loop is None or self.loop.is_closed():
                    self.loop = asyncio.new_event_loop()

                if self.thread is None:
                    self.thread = threading.Thread(target = start_loop, args = (self.loop,))
                    self.thread.start()

                future = asyncio.run_coroutine_threadsafe(self._run_optimization_async(vars_to_estimate, max_iter, disp), self.loop)
                future.add_done_callback(on_completion)

        return self.is_running
    
    async def _run_optimization_async(self, *args, **kwargs):
        func = partial(self.optimize, *args, **kwargs)
        with ThreadPoolExecutor() as pool:
            await self.loop.run_in_executor(pool, func)

    def get_optimization_finished(self):
        return self._optimization_finished
    
    def reset_optimization_loop(self):
        self._optimization_finished = False
        
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread.is_alive():
            self.thread.join()

        self.loop.stop()
        self.loop.close()

        del(self.thread)
        del(self.loop)

        self.thread = None
        self.loop = None

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
