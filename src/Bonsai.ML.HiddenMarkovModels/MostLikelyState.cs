using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Newtonsoft.Json;
using System.Xml.Serialization;
using Python.Runtime;

namespace Bonsai.ML.HiddenMarkovModels
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class MostLikelyState
    {

        private int state;

        /// <summary>
        /// The most likely state of the HMM model.
        /// </summary>
        [JsonProperty("state")]
        [Description("The most likely state of the model.")]
        public int State
        { 
            get => state; 
            set => state = value;
        }


        private int numStates;
        
        /// <summary>
        /// The total number of possible states.
        /// </summary>
        [JsonProperty("num_states")]
        [Description("The total number of possible states.")]
        public int NumStates 
        {
            get => numStates;
            set => numStates = value;
        }

        public IObservable<MostLikelyState> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var statePyObj = pyObject.GetAttr<int>("state");
                var numStatesPyObj = pyObject.GetAttr<int>("num_states");

                return new MostLikelyState {
                    State = statePyObj,
                    NumStates = numStatesPyObj
                };
            });
        }
    }
}