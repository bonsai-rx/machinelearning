using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels
{

    /// <summary>
    /// State of a Hidden Markov Model (HMM).
    /// </summary>
    [Combinator]
    [Description("State of a Hidden Markov Model (HMM).")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class State
    {

        private double[] initStateDistribution;
        private double[,] transitionMatrix;
        private double[,] observationMeans;
        private double[,,] observationCovs;

        /// <summary>
        /// The initial state distribution.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("init_state_distn")]
        [Description("The initial state distribution.")]
        [Category("ModelState")]
        public double[] InitStateDistribution 
        { 
            get => initStateDistribution; 
            set => initStateDistribution = value; 
        }
    
        /// <summary>
        /// The state transition matrix.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("transitions")]
        [Description("The state transition matrix.")]
        [Category("ModelState")]
        public double[,] TransitionMatrix 
        { 
            get => transitionMatrix; 
            set => transitionMatrix = value;
        }

        /// <summary>
        /// The observation means.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_means")]
        [Description("The observation means.")]
        [Category("ModelState")]
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
        [Category("ModelState")]
        public double[,,] ObservationCovs
        {
            get => observationCovs;
            set => observationCovs = value;
        }

        public IObservable<State> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, pyObject =>
            {
                return new State ()
                {
                    InitStateDistribution = InitStateDistribution,
                    TransitionMatrix = TransitionMatrix,
                    ObservationMeans = ObservationMeans,
                    ObservationCovs = ObservationCovs
                };
            });
        }

        public IObservable<State> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var initStateDistributionPyObj = (double[])pyObject.GetArrayAttr("init_state_distn");
                var transitionMatrixPyObj = (double[,])pyObject.GetArrayAttr("transitions");
                var observationMeansPyObj = (double[,])pyObject.GetArrayAttr("observation_means");
                var observationCovsPyObj = (double[,,])pyObject.GetArrayAttr("observation_covs");

                return new State ()
                {
                    InitStateDistribution = initStateDistributionPyObj,
                    TransitionMatrix = transitionMatrixPyObj,
                    ObservationMeans = observationMeansPyObj,
                    ObservationCovs = observationCovsPyObj
                };
            });
        }
    }
}
