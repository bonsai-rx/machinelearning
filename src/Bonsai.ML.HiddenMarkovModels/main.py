from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import asyncio
from ssm import HMM, util
import numpy as np
import autograd.numpy.random as npr
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
import pickle

npr.seed(0)


class HiddenMarkovModel(HMM):

    def __init__(
        self,
        num_states: int,
        dimensions: int,
        observations_model_type: str,
        transitions_model_type: str,
        initial_state_distribution: list[float] = None,
        observations_params: tuple = None,
        observations_kwargs: dict = None,
        transitions_params: tuple = None,
        transitions_kwargs: dict = None
    ):

        self.num_states = num_states
        self.dimensions = dimensions
        self.observations_model_type = observations_model_type
        self.transitions_model_type = transitions_model_type

        if observations_kwargs is not None:
            for (key, value) in observations_kwargs.items():
                if isinstance(value, list):
                    observations_kwargs[key] = np.array(value)

        if transitions_kwargs is not None:
            for (key, value) in transitions_kwargs.items():
                if isinstance(value, list) and key != "hidden_layer_sizes":
                    transitions_kwargs[key] = np.array(value)
                if key == "hidden_layer_sizes":
                    transitions_kwargs[key] = tuple(value)

            if "nonlinearity_type" in transitions_kwargs.keys():
                transitions_kwargs["nonlinearity"] = value
                transitions_kwargs.pop(key)

        super(HiddenMarkovModel, self).__init__(
            K=self.num_states, 
            D=self.dimensions, 
            observations=self.observations_model_type, 
            observation_kwargs=observations_kwargs,
            transitions=self.transitions_model_type,
            transition_kwargs=transitions_kwargs
        )

        self.update_params(initial_state_distribution,
                           transitions_params, observations_params)
        
        if self.transitions_model_type == "nn_recurrent":
            hidden_layer_sizes = np.array([len(layer) for layer in self.transitions.weights[1:]])
            self.transitions.hidden_layer_sizes = hidden_layer_sizes

            def get_nonlinearity_type(func):
                if func == util.relu:
                    return "relu"
                elif func == util.logistic:
                    return "sigmoid"
                else:
                    return "tanh"
                
            self.transitions.nonlinearity_type = get_nonlinearity_type(self.transitions.nonlinearity)

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
        self.predicted_states = np.array([], dtype=int)

    def update_params(self, initial_state_distribution, transitions_params, observations_params):
        hmm_params = self.params

        if initial_state_distribution is not None:
            hmm_params = ((np.array(initial_state_distribution),),
                          ) + hmm_params[1:]

        if transitions_params is not None:
            trans_params = tuple([np.array(param) for param in transitions_params])
            if isinstance(hmm_params[1], tuple):
                hmm_params = (hmm_params[0],) + (trans_params,) + (hmm_params[2],)
            else:
                hmm_params = (hmm_params[0],) + trans_params + (hmm_params[2],)

        if observations_params is not None:
            obs_params = tuple([np.array(param) for param in observations_params])
            if isinstance(hmm_params[2], tuple):
                hmm_params = hmm_params[:2] + (obs_params,)
            else:
                hmm_params = hmm_params[:2] + obs_params

        self.params = hmm_params

        self.initial_state_distribution = hmm_params[0][0]

        if isinstance(hmm_params[1], tuple):
            self.transitions_params = hmm_params[1] 
        else:
            self.transitions_params = (hmm_params[1],)

        if isinstance(hmm_params[2], tuple):
            self.observations_params = hmm_params[2] 
        else:
            self.observations_params = (hmm_params[2],)

    def infer_state(self, observation: list[float]):

        observation = np.expand_dims(np.array(observation), 0)
        self.log_alpha = self.compute_log_alpha(observation, self.log_alpha)
        self.state_probabilities = np.exp(self.log_alpha).astype(np.double)
        prediction = self.state_probabilities.argmax()
        self.predicted_states = np.append(self.predicted_states, prediction)
        self.batch_observations = np.vstack([self.batch_observations, observation])
        return prediction

    def compute_log_alpha(self, obs, log_alpha=None):

        if log_alpha is None:
            log_alpha = (np.log(self.init_state_distn.initial_state_distn) +
                         self.observations.log_likelihoods(obs, None, None, None)).squeeze()
            return log_alpha - logsumexp(log_alpha)

        m = np.max(log_alpha)

        log_alpha = (np.log(np.dot(np.exp(log_alpha - m), self.transitions.transition_matrices(obs, None, None, None).squeeze())
                            ) + m + self.observations.log_likelihoods(obs, None, None, None)).squeeze()

        return log_alpha - logsumexp(log_alpha)
    
    def save_model_to_pickle(self, path: str):
        pickle.save(self, path)

    def load_model_from_pickle(self, path: str):
        self = pickle.load(path)

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

        if not self.is_running and self.loop is None and self.thread is None:

            if self.curr_batch_size >= batch_size:

                if vars_to_estimate is None or vars_to_estimate == {}:
                    vars_to_estimate = {
                        "initial_state_distribution": True,
                        "transitions_params": True,
                        "observations_params": True
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

                    if self.observations_model_type == "gaussian":
                        permutation = calculate_permutation(
                            self.observations_params[0], self.params[2][0])
                        super(HiddenMarkovModel, self).permute(permutation)

                    initial_state_distribution = None if vars_to_estimate[
                        "initial_state_distribution"] else self.initial_state_distribution
                    transitions_params = None if vars_to_estimate[
                        "transitions_params"] else self.transitions_params
                    observations_params = None if vars_to_estimate[
                        "observations_params"] else self.observations_params

                    self.update_params(initial_state_distribution,
                                       transitions_params, observations_params)

                    self.is_running = False
                    self._fit_finished = True
                    self.curr_batch_size = 0

                    if self.flush_data_between_batches:
                        self.batch = None

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
