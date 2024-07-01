using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Bonsai.ML.HiddenMarkovModels.Observations;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

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
        private ObservationParams observations;
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
        [JsonConverter(typeof(StringEnumConverter))]
        [Description("The observation type.")]
        [Category("ModelStateParameters")]
        public ObservationType ObservationType
        {
            get => observationTypeEnum;
            set => observationTypeEnum = value;
        }

        /// <summary>
        /// The observations.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_params")]
        [Description("The observations.")]
        [Category("ModelStateParameters")]
        public ObservationParams Observations
        {
            get => observations;
            set => observations = value;
        }

        public IObservable<StateParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, pyObject =>
            {
                return new StateParameters()
                {
                    InitialStateDistribution = InitialStateDistribution,
                    LogTransitionProbabilities = LogTransitionProbabilities,
                    ObservationType = ObservationType,
                    Observations = Observations
                };
            });
        }

        public IObservable<StateParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var initialStateDistributionPyObj = (double[])pyObject.GetArrayAttr("initial_state_distribution");
                var logTransitionProbabilitiesPyObj = (double[,])pyObject.GetArrayAttr("log_transition_probabilities");
                var observationsPyObj = (object[])pyObject.GetArrayAttr("observation_params");
                var observationTypePyObj = pyObject.GetAttr<string>("observation_type");

                ObservationType = GetFromString(observationTypePyObj);

                var observationClassType = GetObservationsClassType(ObservationType);
                Observations = (ObservationParams)Activator.CreateInstance(observationClassType);
                observations.Params = observationsPyObj;

                return new StateParameters()
                {
                    InitialStateDistribution = initialStateDistributionPyObj,
                    LogTransitionProbabilities = logTransitionProbabilitiesPyObj,
                    ObservationType = ObservationType,
                    Observations = observations
                };
            });
        }

        public override string ToString()
        {
            return $"initial_state_distribution={NumpyHelper.NumpyParser.ParseArray(InitialStateDistribution)},log_transition_probabilities={NumpyHelper.NumpyParser.ParseArray(LogTransitionProbabilities)},{Observations}";
        }
    }
}
