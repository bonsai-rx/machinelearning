using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.HiddenMarkovModels.Observations;
using Bonsai.ML.HiddenMarkovModels.Transitions;
using Bonsai.ML.Python;
using Bonsai.ML.Data;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Represents the state parameters of a Hidden Markov Model (HMM).
    /// </summary>
    [Combinator]
    [JsonConverter(typeof(StateParametersJsonConverter))]
    [Description("State parameters of a Hidden Markov Model (HMM).")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class StateParameters : PythonStringBuilder
    {

        private double[] initialStateDistribution = null;
        private TransitionModel transitions = null;
        private ObservationModel observations = null;

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
            set
            {
                initialStateDistribution = value;
                UpdateString();
            }
        }

        /// <summary>
        /// The transitions model.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("transitions_params")]
        [Description("The transitions model.")]
        [Category("ModelStateParameters")]
        public TransitionModel Transitions
        {
            get => transitions;
            set
            {
                transitions = value;
                UpdateString();
            }
        }

        /// <summary>
        /// The observations.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_params")]
        [Description("The observation model.")]
        [Category("ModelStateParameters")]
        public ObservationModel Observations
        {
            get => observations;
            set
            {
                observations = value;
                UpdateString();
            }
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="StateParameters"/> objects.
        /// </summary>
        public IObservable<StateParameters> Process()
        {
            return Observable.Return(
                new StateParameters()
                {
                    InitialStateDistribution = InitialStateDistribution,
                    Transitions = Transitions,
                    Observations = Observations
                }
            );
        }

        /// <summary>
        /// Takes an observable seqence and returns an observable sequence of <see cref="StateParameters"/> 
        /// objects that are emitted every time the input sequence emits a new element.
        /// </summary>
        public IObservable<StateParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, pyObject =>
            {
                return new StateParameters()
                {
                    InitialStateDistribution = InitialStateDistribution,
                    Transitions = Transitions,
                    Observations = Observations
                };
            });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="StateParameters"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<StateParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var initialStateDistributionPyObj = (double[])pyObject.GetArrayAttr("initial_state_distribution");

                var transitionModelTypePyObj = pyObject.GetAttr<string>("transition_model_type");
                var transitionsParamsPyObj = (Array)pyObject.GetArrayAttr("transitions_params");
                var transitionsParams = (object[])transitionsParamsPyObj;

                var transitionModelType = TransitionModelLookup.GetFromString(transitionModelTypePyObj);
                var transitionsClassType = TransitionModelLookup.GetTransitionsClassType(transitionModelType);
                var transitionsKwargsProperty = transitionsClassType.GetProperty("KwargsArray");

                object[] transitionsConstructorArgs = null;
                if (transitionsKwargsProperty is not null)
                {
                    var transitionsConstructorKeys = (string[])transitionsKwargsProperty.GetValue(null);
                    var transitionsConstructorKeysCount = transitionsConstructorKeys.Length;
                    if (transitionsConstructorKeysCount > 0)
                    {
                        transitionsConstructorArgs = new object[transitionsConstructorKeysCount];
                        var transitionsPyObj = pyObject.GetAttr("transitions");
                        for (int i = 0; i < transitionsConstructorKeysCount; i++)
                        {
                            transitionsConstructorArgs[i] = transitionsPyObj.GetArrayAttr(transitionsConstructorKeys[i]);
                        }
                    }
                }

                transitions = (TransitionModel)Activator.CreateInstance(transitionsClassType, transitionsConstructorArgs);
                transitions.Params = transitionsParams;

                var observationModelTypePyObj = pyObject.GetAttr<string>("observation_model_type");
                var observationsParamsPyObj = (Array)pyObject.GetArrayAttr("observations_params");
                var observationsParams = (object[])observationsParamsPyObj;

                var observationModelType = ObservationModelLookup.GetFromString(observationModelTypePyObj);
                var observationsClassType = ObservationModelLookup.GetObservationsClassType(observationModelType);

                var observationsKwargsProperty = observationsClassType.GetProperty("KwargsArray");

                object[] observationsConstructorArgs = null;
                if (observationsKwargsProperty is not null)
                {
                    var observationsConstructorKeys = (string[])observationsKwargsProperty.GetValue(null);
                    var observationsConstructorKeysCount = observationsConstructorKeys.Length;
                    if (observationsConstructorKeysCount > 0)
                    {
                        observationsConstructorArgs = new object[observationsConstructorKeysCount];
                        var observationsPyObj = pyObject.GetAttr("observations");
                        for (int i = 0; i < observationsConstructorKeysCount; i++)
                        {
                            observationsConstructorArgs[i] = observationsPyObj.GetArrayAttr(observationsConstructorKeys[i]);
                        }
                    }
                }

                observations = (ObservationModel)Activator.CreateInstance(observationsClassType, observationsConstructorArgs);
                observations.Params = observationsParams;

                return new StateParameters()
                {
                    InitialStateDistribution = initialStateDistributionPyObj,
                    Transitions = transitions,
                    Observations = observations
                };
            });
        }

        /// <inheritdoc/>
        protected override string BuildString()
        {
            StringBuilder.Clear();

            if (InitialStateDistribution != null)
            {
                StringBuilder.Append($"initial_state_distribution={StringFormatter.FormatToPython(InitialStateDistribution)},");
            }

            if (Transitions != null)
            {
                StringBuilder.Append($"{Transitions},");
            }

            if (Observations != null)
            {
                StringBuilder.Append($"{Observations},");
            }

            if (StringBuilder.Length > 0)
            {
                StringBuilder.Remove(StringBuilder.Length - 1, 1);
            }

            return StringBuilder.ToString();
        }
    }
}
