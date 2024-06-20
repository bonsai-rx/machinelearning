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
    [WorkflowElementCategory(ElementCategory.Source)]
    public class ModelParameters
    {

        private int numStates;
        private string numStatesStr = "";

        /// <summary>
        /// The number of states of the HMM model.
        /// </summary>
        [JsonProperty("num_states")]
        [Description("The number of discrete latent states of the HMM model")]
        [Category("InitialParameters")]
        public int NumStates 
        { 
            get => numStates; 
            set 
            { 
                numStates = value; 
                numStatesStr = numStates.ToString(); 
            } 
        }


        private int dimensions;
        private string dimensionsStr = "";
        
        /// <summary>
        /// The dimensionality of the observations into the HMM model.
        /// </summary>
        [JsonProperty("dimensions")]
        [Description("The dimensionality of the observations into the HMM model")]
        [Category("InitialParameters")]
        public int Dimensions 
        {
            get => dimensions;
            set 
            {
                dimensions = value;
                dimensionsStr = dimensions.ToString(); 
            } 
        }


        private ObservationType observationType;
        private string observationTypeStr = "";

        /// <summary>
        /// The type of distribution that the HMM will use to model the emission of data observations.
        /// </summary>
        [JsonProperty("observation_type")]
        [Description("The type of distribution that the HMM will use to model the emission of data observations.")]
        [Category("InitialParameters")]
        public ObservationType ObservationType 
        { 
            get => observationType; 
            set 
            { 
                observationType = value; 
                observationTypeLookup.TryGetValue(observationType, out observationTypeStr); 
            } 
        }


        private StateParameters stateParameters = null;
        private string stateParametersStr = "";

        /// <summary>
        /// The state parameters of the HMM model.
        /// </summary>
        [XmlIgnore]
        [Description("The state parameters of the HMM model.")]
        [Category("ModelState")]
        public StateParameters StateParameters
        {
            get => stateParameters;
            set 
            {
                stateParameters = value;
                stateParametersStr = stateParameters == null ? "" : 
                    $"initial_state_distribution={NumpyHelper.NumpyParser.ParseArray(stateParameters.InitialStateDistribution)}," +
                    $"log_transition_probabilities={NumpyHelper.NumpyParser.ParseArray(stateParameters.LogTransitionProbabilities)}," +
                    $"observation_means={NumpyHelper.NumpyParser.ParseArray(stateParameters.ObservationMeans)}," +
                    $"observation_covs={NumpyHelper.NumpyParser.ParseArray(stateParameters.ObservationCovs)}";
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelParameters"/> class.
        /// </summary>
        public ModelParameters()
        {
            NumStates = 2;
            Dimensions = 2;
            ObservationType = ObservationType.Gaussian;
        }

        public IObservable<ModelParameters> Process()
        {
            return Observable.Return(
                new ModelParameters()  
                { 
                    NumStates = NumStates, 
                    Dimensions = Dimensions, 
                    ObservationType = ObservationType, 
                    StateParameters = StateParameters
                });
        }

        public IObservable<ModelParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, item => 
            {
                return new ModelParameters() 
                { 
                    NumStates = NumStates, 
                    Dimensions = Dimensions, 
                    ObservationType = ObservationType, 
                    StateParameters = StateParameters
                };
            });
        }

        public IObservable<ModelParameters> Process(IObservable<PyObject> source)
        {
            var sharedSource = source.Publish().RefCount();
            var stateParametersObservable = new StateParameters().Process(sharedSource);
            return sharedSource.Select(pyObject => 
            {
                var numStatesPyObj = pyObject.GetAttr<int>("num_states");
                var dimensionsPyObj = pyObject.GetAttr<int>("dimensions");
                var observationTypeStrPyObj = pyObject.GetAttr<string>("observation_type");

                observationTypeStrLookup.TryGetValue(observationTypeStrPyObj, out var observationTypePyObj);

                return new ModelParameters() 
                { 
                    NumStates = numStatesPyObj, 
                    Dimensions = dimensionsPyObj, 
                    ObservationType = observationTypePyObj, 
                };
            }).Zip(stateParametersObservable, (modelParameters, stateParameters) => 
            {
                modelParameters.StateParameters = stateParameters;
                return modelParameters;
            });
        }

        public override string ToString()
        {
            return $"num_states={numStatesStr}, dimensions={dimensionsStr}, observation_type=\"{observationTypeStr}\", {stateParametersStr}";
        }

        private static readonly Dictionary<ObservationType, string> observationTypeLookup = new Dictionary<ObservationType, string>
        {
            { ObservationType.Gaussian, "gaussian" },
            { ObservationType.Exponential, "exponential" },
            { ObservationType.Poisson, "poisson" },
            { ObservationType.Bernoulli, "bernoulli" },
            { ObservationType.Autoregressive, "ar" }
        };

        private static readonly Dictionary<string, ObservationType> observationTypeStrLookup = new Dictionary<string, ObservationType>
        {
            { "gaussian", ObservationType.Gaussian },
            { "exponential", ObservationType.Exponential },
            { "poisson", ObservationType.Poisson },
            { "bernoulli", ObservationType.Bernoulli },
            { "ar", ObservationType.Autoregressive }
        };
    }
}

