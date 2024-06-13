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

    def infer_state(self, observation: list[float]):

        self.log_alpha = self.compute_log_alpha(np.expand_dims(np.array(observation), 0), self.log_alpha)
        self.state_probabilities = np.exp(self.log_alpha).astype(np.double)

    def compute_log_alpha(self, obs, log_alpha = None):
        
        if log_alpha is None:
            log_alpha = (np.log(self.init_state_distn.initial_state_distn) + self.observations.log_likelihoods(obs, None, None, None)).squeeze()
            return log_alpha - logsumexp(log_alpha)
        
        m = np.max(log_alpha)
        log_alpha = (np.log(np.dot(np.exp(log_alpha - m), self.transitions.transition_matrix)) + m + self.observations.log_likelihoods(obs, None, None, None)).squeeze()
        return log_alpha - logsumexp(log_alpha)
