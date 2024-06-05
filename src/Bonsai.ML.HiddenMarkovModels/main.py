from ssm import HMM
import numpy as np
import autograd.numpy.random as npr
npr.seed(0)

class HiddenMarkovModel(HMM):

    def __init__(self, num_states, dimensions, observation_type, init_state_distn = None, transitions = None, observations = None):
        self.num_states = num_states
        self.dimensions = dimensions
        self.observation_type = observation_type
        super(HiddenMarkovModel, self).__init__(K=self.num_states, D=self.dimensions, observations=self.observation_type)

        hmm_params = self.params

        if init_state_distn is not None:
            hmm_params = ((np.array(init_state_distn),),) + hmm_params[1:]

        if transitions is not None:
            hmm_params = (hmm_params[0],) + ((np.array(transitions),),) + (hmm_params[2],)
        
        if observations is not None:
            hmm_params = hmm_params[:2] + ((np.array(observations),),)

        self.params = hmm_params

        self.init_state_distn = self.params[0][0]
        self.transitions = self.params[1][0]
        self.observations = list(self.params[2])

    def most_likely_state(self, observation):
        self.state = super(HiddenMarkovModel, self).most_likely_state(observation)