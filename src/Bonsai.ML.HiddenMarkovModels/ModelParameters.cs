using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using System.Text;
using Newtonsoft.Json;
using System.Xml.Serialization;
using Python.Runtime;
using Bonsai.ML.HiddenMarkovModels.Observations;
using Bonsai.ML.HiddenMarkovModels.Transitions;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Represents the model parameters of a Hidden Markov Model (HMM).
    /// </summary>
    [Combinator]
    [Description("Model parameters of a Hidden Markov Model (HMM).")]
    [WorkflowElementCategory(ElementCategory.Source)]
    [JsonConverter(typeof(ModelParametersJsonConverter))]
    public class ModelParameters : PythonStringBuilder
    {

        private int numStates;
        private int dimensions;
        private ObservationsType observationsType;
        private TransitionsType transitionsType;
        private StateParameters stateParameters = null;

        /// <summary>
        /// The number of states of the HMM model.
        /// </summary>
        [Description("The number of discrete latent states of the HMM model")]
        [Category("ModelSpecification")]
        public int NumStates { get => numStates; set { numStates = value; UpdateString(); } }

        /// <summary>
        /// The dimensionality of the observations into the HMM model.
        /// </summary>
        [Description("The dimensionality of the observations into the HMM model")]
        [Category("ModelSpecification")]
        public int Dimensions { get => dimensions; set { dimensions = value; UpdateString(); } }

        /// <summary>
        /// The type of distribution that the HMM will use to model the emission of data observations.
        /// </summary>
        [Description("The type of distribution that the HMM will use to model the emission of data observations.")]
        [Category("ModelSpecification")]
        public ObservationsType ObservationsType { get => observationsType; set { observationsType = value; UpdateString(); } }

        /// <summary>
        /// The type of transition model that the HMM will use to calculate the probabilities of transitioning between states.
        /// </summary>
        [Description("The type of transition model that the HMM will use to calculate the probabilities of transitioning between states.")]
        [Category("ModelSpecification")]
        public TransitionsType TransitionsType { get => transitionsType; set { transitionsType = value; UpdateString(); } }

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
                if (value != null)
                {
                    if (stateParameters.Observations != null)
                    {
                        ObservationsType = stateParameters.Observations.ObservationsType;
                    }
                    if (stateParameters.Transitions != null)
                    {
                        TransitionsType = stateParameters.Transitions.TransitionsType;
                    }
                }
                UpdateString();
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelParameters"/> class.
        /// </summary>
        public ModelParameters()
        {
            NumStates = 2;
            Dimensions = 2;
            ObservationsType = ObservationsType.Gaussian;
            TransitionsType = TransitionsType.Stationary;
        }

        /// <summary>
        /// Returns an observable sequence of <see cref="ModelParameters"/> objects.
        /// </summary>
        public IObservable<ModelParameters> Process()
        {
            return Observable.Return(
                new ModelParameters()
                {
                    NumStates = NumStates,
                    Dimensions = Dimensions,
                    ObservationsType = ObservationsType,
                    TransitionsType = TransitionsType,
                    StateParameters = StateParameters
                });
        }

        /// <summary>
        /// Takes an observable seqence and returns an observable sequence of <see cref="ModelParameters"/> 
        /// objects that are emitted every time the input sequence emits a new element.
        /// </summary>
        public IObservable<ModelParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, item =>
            {
                return new ModelParameters()
                {
                    NumStates = NumStates,
                    Dimensions = Dimensions,
                    ObservationsType = ObservationsType,
                    TransitionsType = TransitionsType,
                    StateParameters = StateParameters
                };
            });
        }

        /// <summary>
        /// Transforms an observable sequence of <see cref="PyObject"/> into an observable sequence 
        /// of <see cref="ModelParameters"/> objects by accessing internal attributes of the <see cref="PyObject"/>.
        /// </summary>
        public IObservable<ModelParameters> Process(IObservable<PyObject> source)
        {
            var sharedSource = source.Publish().RefCount();
            var stateParametersObservable = new StateParameters().Process(sharedSource);
            return sharedSource.Select(pyObject =>
            {
                numStates = pyObject.GetAttr<int>("num_states");
                dimensions = pyObject.GetAttr<int>("dimensions");
                var observationsTypeStrPyObj = pyObject.GetAttr<string>("observation_type");
                var transitionsTypeStrPyObj = pyObject.GetAttr<string>("transition_type");

                observationsType = ObservationsLookup.GetFromString(observationsTypeStrPyObj);
                transitionsType = TransitionsLookup.GetFromString(transitionsTypeStrPyObj);

                return new ModelParameters()
                {
                    NumStates = NumStates,
                    Dimensions = Dimensions,
                    ObservationsType = ObservationsType,
                    TransitionsType = TransitionsType
                };
            }).Zip(stateParametersObservable, (modelParameters, stateParameters) =>
            {
                modelParameters.StateParameters = stateParameters;
                return modelParameters;
            });
        }

        /// <inheritdoc/>
        protected override string BuildString()
        {
            StringBuilder.Clear();
            StringBuilder.Append($"num_states={numStates},")
                .Append($"dimensions={dimensions},");
            if (stateParameters == null || string.IsNullOrEmpty(stateParameters.ToString()))
            {
                StringBuilder.Append($"observation_type=\"{ObservationsLookup.GetString(observationsType)}\",");
                StringBuilder.Append($"transition_type=\"{TransitionsLookup.GetString(transitionsType)}\"");
            }
            else
            {
                StringBuilder.Append($"{stateParameters},");
                if (stateParameters.Observations == null)
                {
                    StringBuilder.Append($"observation_type=\"{ObservationsLookup.GetString(observationsType)}\",");
                }
                if (stateParameters.Transitions == null)
                {
                    StringBuilder.Append($"transition_type=\"{TransitionsLookup.GetString(transitionsType)}\",");
                }
                StringBuilder.Remove(StringBuilder.Length - 1, 1);
            }
            return StringBuilder.ToString();
        }
    }
}

