from ssm import HMM
import numpy as np
import autograd.numpy.random as npr
npr.seed(0)

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelState(BaseModel):
    init_state_distn: List[float] = None
    transitions: List[List[float]] = None
    observations: List[List[float]] = None


@dataclass
class InitParams(BaseModel):
    num_states: int
    dimensions: int
    observation_type: str


@dataclass
class State(BaseModel):
    state: int
    num_states: int


class HiddenMarkovModel(HMM):

    def __init__(self, init_params: InitParams, model_state: ModelState):
        num_states = init_params.num_states
        dimensions = init_params.dimensions
        observation_type = init_params.observation_type
        super(HiddenMarkovModel, self).__init__(K=num_states, D=dimensions, observations=observation_type)

        hmm_params = self.params

        if model_state.init_state_distn is not None:
            hmm_params = ((np.array(model_state.init_state_distn),),) + hmm_params[1:]

        if model_state.transitions is not None:
            hmm_params = (hmm_params[0],) + ((np.array(model_state.transitions),),) + (hmm_params[2],)
        
        if model_state.observations is not None:
            hmm_params = hmm_params[:2] + ((np.array(model_state.observations),),)

        self.params = hmm_params

        init_state_distn = self.params[0][0]
        transitions = self.params[1][0]
        observations = list(self.params[2])

        self.init_params = init_params
        self.model_state = ModelState(init_state_distn, transitions, observations)

    def most_likely_state(self, observation: List[float]):
        return State(super(HiddenMarkovModel, self).most_likely_state(observation), self.num_states)