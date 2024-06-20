from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import asyncio
from ssm import HMM
from ssm.messages import logsumexp
import numpy as np
import autograd.numpy.random as npr

npr.seed(0)


class HiddenMarkovModel(HMM):

    def __init__(
        self,
        num_states,
        dimensions,
        observation_type,
        initial_state_distribution=None,
        log_transition_probabilities=None,
        observation_means=None,
        observation_covs=None,
    ):

        self.num_states = num_states
        self.dimensions = dimensions
        self.observation_type = observation_type
        super(HiddenMarkovModel, self).__init__(
            K=self.num_states, D=self.dimensions, observations=self.observation_type
        )

        hmm_params = self.params

        if initial_state_distribution is not None:
            hmm_params = ((np.array(initial_state_distribution),),
                          ) + hmm_params[1:]

        if log_transition_probabilities is not None:
            hmm_params = (
                (hmm_params[0],) +
                ((np.array(log_transition_probabilities),),) + (hmm_params[2],)
            )

        if observation_means is not None and observation_covs is not None:
            hmm_params = hmm_params[:2] + (
                (np.array(observation_means), np.array(observation_covs)),
            )

        self.params = hmm_params

        self.initial_state_distribution = hmm_params[0][0]
        self.log_transition_probabilities = hmm_params[1][0]
        self.observation_means = hmm_params[2][0]
        self.observation_stds = np.diagonal(hmm_params[2][1], axis1=1, axis2=2)
        self.observation_covs = hmm_params[2][1]

        self.log_alpha = None
        self.state_probabilities = None

        self.batch = None
        self.is_running = False
        self._fit_finished = False
        self.loop = None
        self.thread = None
        self.curr_batch_size = 0
        self.flush_data_between_batches = True
        self.most_likely_state_sequence = None

    def infer_state(self, observation: list[float]):

        self.log_alpha = self.compute_log_alpha(
            np.expand_dims(np.array(observation), 0), self.log_alpha)
        self.state_probabilities = np.exp(self.log_alpha).astype(np.double)

    def compute_log_alpha(self, obs, log_alpha=None):

        if log_alpha is None:
            log_alpha = (np.log(self.init_state_distn.initial_state_distn) +
                         self.observations.log_likelihoods(obs, None, None, None)).squeeze()
            return log_alpha - logsumexp(log_alpha)

        m = np.max(log_alpha)
        log_alpha = (np.log(np.dot(np.exp(log_alpha - m), self.transitions.transition_matrix)
                            ) + m + self.observations.log_likelihoods(obs, None, None, None)).squeeze()
        return log_alpha - logsumexp(log_alpha)

    def fit_async(self,
                  observation: list[float],
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

        if not self.is_running:

            if self.curr_batch_size >= batch_size:

                def start_loop(loop):
                    asyncio.set_event_loop(loop)
                    loop.run_forever()

                def on_completion(future):

                    self.initial_state_distribution = self.params[0][0]
                    self.log_transition_probabilities = self.params[1][0]
                    self.observation_means = self.params[2][0]
                    self.observation_stds = np.diagonal(self.params[2][1], axis1=1, axis2=2)
                    self.observation_covs = self.params[2][1]

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

    def most_likely_state_seq_from_batch(self):
        if self.batch is not None:
            self.most_likely_state_sequence = self.most_likely_states(self.batch)
