from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import asyncio
from ssm import HMM
import numpy as np
import autograd.numpy.random as npr
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

npr.seed(0)


class HiddenMarkovModel(HMM):

    def __init__(
        self,
        num_states: int,
        dimensions: int,
        observation_type: str,
        transition_type: str,
        initial_state_distribution: list[float] = None,
        observation_params: tuple = None,
        observation_kwargs: dict = None,
        transition_params: tuple = None,
        transition_kwargs: dict = None
    ):

        self.num_states = num_states
        self.dimensions = dimensions
        self.observation_type = observation_type
        self.transition_type = transition_type

        if observation_kwargs is not None:
            for (key, value) in observation_kwargs.items():
                if isinstance(value, list):
                    observation_kwargs[key] = np.array(value)

        if transition_kwargs is not None:
            for (key, value) in transition_kwargs.items():
                if isinstance(value, list):
                    transition_kwargs[key] = np.array(value)

        super(HiddenMarkovModel, self).__init__(
            K=self.num_states, 
            D=self.dimensions, 
            observations=self.observation_type, 
            observation_kwargs=observation_kwargs,
            transitions=transition_type,
            transition_kwargs=transition_kwargs
        )

        self.update_params(initial_state_distribution,
                           transition_params, observation_params)
        
        if observation_kwargs is None:
            observation_kwargs = {}

        self.observation_kwargs = observation_kwargs

        if transition_kwargs is None:
            transition_kwargs = {}

        self.transition_kwargs = transition_kwargs

        self.log_alpha = None
        self.state_probabilities = None

        self.batch = None
        self.batch_observations = np.array([[]], dtype=float)
        self.is_running = False
        self._fit_finished = False
        self.loop = None
        self.thread = None
        self.curr_batch_size = 0
        self.flush_data_between_batches = True
        self.inferred_most_probable_states = np.array([], dtype=int)

    def update_params(self, initial_state_distribution, transition_params, observation_params):
        hmm_params = self.params

        if initial_state_distribution is not None:
            hmm_params = ((np.array(initial_state_distribution),),
                          ) + hmm_params[1:]

        if transition_params is not None:
            trans_params = tuple([np.array(param) for param in transition_params])
            if isinstance(hmm_params[1], tuple):
                hmm_params = (hmm_params[0],) + (trans_params,) + (hmm_params[2],)
            else:
                hmm_params = (hmm_params[0],) + trans_params + (hmm_params[2],)

        if observation_params is not None:
            obs_params = tuple([np.array(param) for param in observation_params])
            if isinstance(hmm_params[2], tuple):
                hmm_params = hmm_params[:2] + (obs_params,)
            else:
                hmm_params = hmm_params[:2] + obs_params

        self.params = hmm_params

        self.initial_state_distribution = hmm_params[0][0]

        if isinstance(hmm_params[1], tuple):
            self.transition_params = hmm_params[1] 
        else:
            self.transition_params = (hmm_params[1],)

        if isinstance(hmm_params[2], tuple):
            self.observation_params = hmm_params[2] 
        else:
            self.observation_params = (hmm_params[2],)

    def infer_state(self, observation: list[float]):

        self.log_alpha = self.compute_log_alpha(
            np.expand_dims(np.array(observation), 0), self.log_alpha)
        self.state_probabilities = np.exp(self.log_alpha).astype(np.double)
        return self.state_probabilities.argmax()

    def compute_log_alpha(self, obs, log_alpha=None):

        if log_alpha is None:
            log_alpha = (np.log(self.init_state_distn.initial_state_distn) +
                         self.observations.log_likelihoods(obs, None, None, None)).squeeze()
            return log_alpha - logsumexp(log_alpha)

        m = np.max(log_alpha)

        log_alpha = (np.log(np.dot(np.exp(log_alpha - m), self.transitions.transition_matrices(obs, None, None, None).squeeze())
                            ) + m + self.observations.log_likelihoods(obs, None, None, None)).squeeze()

        return log_alpha - logsumexp(log_alpha)

    def fit_async(self,
                  observation: list[float],
                  vars_to_estimate: dict = None,
                  batch_size: int = 20,
                  max_iter: int = 50,
                  flush_data_between_batches: bool = False):

        self.flush_data_between_batches = flush_data_between_batches

        if self.batch is None:
            self.batch = np.expand_dims(np.array(observation), 0)
            self.curr_batch_size += 1

        elif self.curr_batch_size < batch_size or not flush_data_between_batches:
            self.batch = np.vstack(
                [self.batch, np.expand_dims(np.array(observation), 0)])
            self.curr_batch_size += 1

        elif self.curr_batch_size == batch_size:
            self.batch = np.vstack(
                [self.batch[1:], np.expand_dims(np.array(observation), 0)])

        self.batch_observations = self.batch

        if not self.is_running and self.loop is None and self.thread is None:

            if self.curr_batch_size >= batch_size:

                if vars_to_estimate is None or vars_to_estimate == {}:
                    vars_to_estimate = {
                        "initial_state_distribution": True,
                        "transition_params": True,
                        "observation_params": True
                    }

                def calculate_permutation(mat1, mat2):
                    num_states = mat1.shape[0]
                    cost_matrix = np.zeros((num_states, num_states))
                    for i in range(num_states):
                        for j in range(num_states):
                            cost_matrix[i, j] = np.linalg.norm(
                                mat1[i] - mat2[j])
                    return linear_sum_assignment(cost_matrix)[1]

                def start_loop(loop):
                    asyncio.set_event_loop(loop)
                    loop.run_forever()

                def on_completion(future):

                    if self.observation_type == "gaussian":
                        permutation = calculate_permutation(
                            self.observation_params[0], self.params[2][0])
                        super(HiddenMarkovModel, self).permute(permutation)

                    initial_state_distribution = None if vars_to_estimate[
                        "initial_state_distribution"] else self.initial_state_distribution
                    transition_params = None if vars_to_estimate[
                        "transition_params"] else self.transition_params
                    observation_params = None if vars_to_estimate[
                        "observation_params"] else self.observation_params

                    self.update_params(initial_state_distribution,
                                       transition_params, observation_params)

                    self.is_running = False
                    self._fit_finished = True
                    self.curr_batch_size = 0

                    if self.flush_data_between_batches:
                        self.batch = None

                    self.inferred_most_probable_states = np.array([self.infer_state(obs) for obs in self.batch_observations]).astype(int)

                self.is_running = True

                if self.loop is None or self.loop.is_closed():
                    self.loop = asyncio.new_event_loop()

                if self.thread is None:
                    self.thread = threading.Thread(
                        target=start_loop, args=(self.loop,))
                    self.thread.start()

                future = asyncio.run_coroutine_threadsafe(self._fit_async(
                    self.batch, method="em", num_iters=max_iter, init_method="kmeans"), self.loop)
                future.add_done_callback(on_completion)

        return self.is_running

    async def _fit_async(self, *args, **kwargs):
        func = partial(super(HiddenMarkovModel, self).fit, *args, **kwargs)
        with ThreadPoolExecutor() as pool:
            await self.loop.run_in_executor(pool, func)

    def get_fit_finished(self):
        return self._fit_finished

    def reset_fit_loop(self):
        self._fit_finished = False

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread.is_alive():
            self.thread.join()

        self.loop.stop()
        self.loop.close()

        del (self.thread)
        del (self.loop)

        self.thread = None
        self.loop = None
