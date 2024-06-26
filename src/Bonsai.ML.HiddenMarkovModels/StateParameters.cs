using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.HiddenMarkovModels.Observations;
using System.Collections.Generic;

namespace Bonsai.ML.HiddenMarkovModels
{

    /// <summary>
    /// StateParameters of a Hidden Markov Model (HMM).
    /// </summary>
    [Combinator]
    [Description("StateParameters of a Hidden Markov Model (HMM).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class StateParameters
    {

        private double[] initialStateDistribution;
        private double[,] logTransitionProbabilities;
        private ObservationParams observationParams;
        private ObservationType observationTypeEnum;

        /// <summary>
        /// The initial state distribution.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("initial_state_distribution")]
        [Description("The initial state distribution.")]
        [Category("ModelStateParameters")]
        public double[] InitialStateDistribution 
        { 
            get => initialStateDistribution; 
            set => initialStateDistribution = value; 
        }
    
        /// <summary>
        /// The log of the state transition probabilities.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("log_transition_probabilities")]
        [Description("The log of the state transition probabilities.")]
        [Category("ModelStateParameters")]
        public double[,] LogTransitionProbabilities 
        { 
            get => logTransitionProbabilities; 
            set => logTransitionProbabilities = value;
        }

        /// <summary>
        /// The observation type.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_type")]
        [Description("The observation type.")]
        [Category("ModelStateParameters")]
        public ObservationType ObservationType
        {
            get => observationTypeEnum;
            set => observationTypeEnum = value;
        }

        /// <summary>
        /// The observation parameters.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_params")]
        [Description("The observation parameters.")]
        [Category("ModelStateParameters")]
        public ObservationParams ObservationParams
        {
            get => observationParams;
            set => observationParams = value;
        }

        public IObservable<StateParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, pyObject =>
            {
                return new StateParameters ()
                {
                    InitialStateDistribution = InitialStateDistribution,
                    LogTransitionProbabilities = LogTransitionProbabilities,
                    ObservationParams = ObservationParams
                };
            });
        }

        public IObservable<StateParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var initialStateDistributionPyObj = (double[])pyObject.GetArrayAttr("initial_state_distribution");
                var logTransitionProbabilitiesPyObj = (double[,])pyObject.GetArrayAttr("log_transition_probabilities");
                var observationParamsPyObj = (object[])pyObject.GetArrayAttr("observation_params");
                observationTypeEnumLookup.TryGetValue(ObservationType, out Type observationType);
                ObservationParams observationParams = (ObservationParams)Activator.CreateInstance(observationType);
                observationParams.Params = observationParamsPyObj;

                return new StateParameters ()
                {
                    InitialStateDistribution = initialStateDistributionPyObj,
                    LogTransitionProbabilities = logTransitionProbabilitiesPyObj,
                    ObservationType = ObservationType,
                    ObservationParams = observationParams
                };
            });
        }

        private static readonly Dictionary<ObservationType, Type> observationTypeEnumLookup = new Dictionary<ObservationType, Type>
        {
            { ObservationType.Gaussian, typeof(GaussianObservations) },
            { ObservationType.Exponential, typeof(ExponentialObservations) },
            { ObservationType.Poisson, typeof(PoissonObservations) },
            { ObservationType.Bernoulli, typeof(BernoulliObservations) },
            { ObservationType.Autoregressive, typeof(AutoRegressiveObservations) }
        };
    }
}
