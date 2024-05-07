from lds.inference import OnlineKalmanFilter
import numpy as np

import lds.learning
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import partial

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