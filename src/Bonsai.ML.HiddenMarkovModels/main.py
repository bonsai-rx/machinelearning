from ssm import HMM
import numpy as np
import autograd.numpy.random as npr
npr.seed(0)

class HiddenMarkovModel(HMM):

    def __init__(self, num_states, dimensions, observation_type, init_state_distribution = None, transition_matrix = None, observation_means = None, observation_covs = None):
        
        self.num_states = num_states
        self.dimensions = dimensions
        self.observation_type = observation_type
        super(HiddenMarkovModel, self).__init__(K=self.num_states, D=self.dimensions, observations=self.observation_type)

        hmm_params = self.params

        if init_state_distribution is not None:
            hmm_params = ((np.array(init_state_distribution),),) + hmm_params[1:]

        if transition_matrix is not None:
            hmm_params = (hmm_params[0],) + ((np.array(transition_matrix),),) + (hmm_params[2],)
        
        if observation_means is not None and observation_covs is not None:
            hmm_params = hmm_params[:2] + ((np.array(observation_means), np.array(observation_covs)),)

        self.params = hmm_params

        self.init_state_distribution = hmm_params[0][0]
        self.transition_matrix = hmm_params[1][0]
        self.observation_means = hmm_params[2][0]
        self.observation_covs = hmm_params[2][1]

    def most_likely_state(self, observation):
        self.state = super(HiddenMarkovModel, self).most_likely_state(observation)