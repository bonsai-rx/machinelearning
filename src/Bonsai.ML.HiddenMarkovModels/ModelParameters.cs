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
using Bonsai.ML.Python;

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
        private ObservationsModelType observationsModelType;
        private TransitionsModelType transitionsModelType;
        private StateParameters stateParameters = null;

        /// <summary>
        /// The number of states of the HMM model.
        /// </summary>
        [Description("The number of discrete latent states of the HMM model.")]
        [Category("ModelSpecification")]
        public int NumStates 
        { 
            get => numStates; 
            set 
            { 
                numStates = value; 
                UpdateString(); 
            } 
        }

        /// <summary>
        /// The dimensionality of the observations into the HMM model.
        /// </summary>
        [Description("The dimensionality of the observations into the HMM model.")]
        [Category("ModelSpecification")]
        public int Dimensions 
        { 
            get => dimensions; 
            set 
            { 
                dimensions = value; 
                UpdateString(); 
            } 
        }

        /// <summary>
        /// The type of distribution that the HMM will use to model the emission of data observations.
        /// </summary>
        [Description("The type of distribution that the HMM will use to model the emission of data observations.")]
        [Category("ModelSpecification")]
        public ObservationsModelType ObservationsModelType 
        { 
            get => observationsModelType; 
            set 
            { 
                observationsModelType = value; 
                UpdateString(); 
            } 
        }

        /// <summary>
        /// The type of transition model that the HMM will use to calculate the probabilities of transitioning between states.
        /// </summary>
        [Description("The type of transition model that the HMM will use to calculate the probabilities of transitioning between states.")]
        [Category("ModelSpecification")]
        public TransitionsModelType TransitionsModelType 
        { 
            get => transitionsModelType; 
            set 
            { 
                transitionsModelType = value; 
                UpdateString(); 
            } 
        }

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
                        ObservationsModelType = stateParameters.Observations.ObservationsModelType;
                    }
                    if (stateParameters.Transitions != null)
                    {
                        TransitionsModelType = stateParameters.Transitions.TransitionsModelType;
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
            ObservationsModelType = ObservationsModelType.Gaussian;
            TransitionsModelType = TransitionsModelType.Stationary;
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
                    ObservationsModelType = ObservationsModelType,
                    TransitionsModelType = TransitionsModelType,
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
                    ObservationsModelType = ObservationsModelType,
                    TransitionsModelType = TransitionsModelType,
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
                var observationsModelTypeStrPyObj = pyObject.GetAttr<string>("observations_model_type");
                var transitionsModelTypeStrPyObj = pyObject.GetAttr<string>("transitions_model_type");

                observationsModelType = ObservationsModelLookup.GetFromString(observationsModelTypeStrPyObj);
                transitionsModelType = TransitionsModelLookup.GetFromString(transitionsModelTypeStrPyObj);

                return new ModelParameters()
                {
                    NumStates = NumStates,
                    Dimensions = Dimensions,
                    ObservationsModelType = ObservationsModelType,
                    TransitionsModelType = TransitionsModelType
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
                StringBuilder.Append($"observations_model_type=\"{ObservationsModelLookup.GetString(observationsModelType)}\",");
                StringBuilder.Append($"transitions_model_type=\"{TransitionsModelLookup.GetString(transitionsModelType)}\"");
            }
            else
            {
                StringBuilder.Append($"{stateParameters},");
                if (stateParameters.Observations == null)
                {
                    StringBuilder.Append($"observations_model_type=\"{ObservationsModelLookup.GetString(observationsModelType)}\",");
                }
                if (stateParameters.Transitions == null)
                {
                    StringBuilder.Append($"transitions_model_type=\"{TransitionsModelLookup.GetString(transitionsModelType)}\",");
                }
                StringBuilder.Remove(StringBuilder.Length - 1, 1);
            }
            return StringBuilder.ToString();
        }
    }
}

