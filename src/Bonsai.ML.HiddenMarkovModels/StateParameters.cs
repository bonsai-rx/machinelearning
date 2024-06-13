using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;

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
        private double[,] observationMeans;
        private double[,,] observationCovs;

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
        /// The observation means.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_means")]
        [Description("The observation means.")]
        [Category("ModelStateParameters")]
        public double[,] ObservationMeans
        {
            get => observationMeans;
            set => observationMeans = value;
        }

        /// <summary>
        /// The observation covariances.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_covs")]
        [Description("The observation covariances.")]
        [Category("ModelStateParameters")]
        public double[,,] ObservationCovs
        {
            get => observationCovs;
            set => observationCovs = value;
        }

        public IObservable<StateParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, pyObject =>
            {
                return new StateParameters ()
                {
                    InitialStateDistribution = InitialStateDistribution,
                    LogTransitionProbabilities = LogTransitionProbabilities,
                    ObservationMeans = ObservationMeans,
                    ObservationCovs = ObservationCovs
                };
            });
        }

        public IObservable<StateParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var initialStateDistributionPyObj = (double[])pyObject.GetArrayAttr("initial_state_distribution");
                var logTransitionProbabilitiesPyObj = (double[,])pyObject.GetArrayAttr("log_transition_probabilities");
                var observationMeansPyObj = (double[,])pyObject.GetArrayAttr("observation_means");
                var observationCovsPyObj = (double[,,])pyObject.GetArrayAttr("observation_covs");

                return new StateParameters ()
                {
                    InitialStateDistribution = initialStateDistributionPyObj,
                    LogTransitionProbabilities = logTransitionProbabilitiesPyObj,
                    ObservationMeans = observationMeansPyObj,
                    ObservationCovs = observationCovsPyObj
                };
            });
        }
    }
}
