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
    public class ModelParams
    {

        private int numStates;
        private string numStatesStr = "";

        /// <summary>
        /// The number of states of the HMM model.
        /// </summary>
        [JsonProperty("num_states")]
        [Description("The number of discrete latent states of the HMM model")]
        [Category("InitParameters")]
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
        [Category("InitParameters")]
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
        [Category("InitParameters")]
        public ObservationType ObservationType 
        { 
            get => observationType; 
            set 
            { 
                observationType = value; 
                observationTypeLookup.TryGetValue(observationType, out observationTypeStr); 
            } 
        }


        private double[] initStateDistribution = null;
        private string initStateDistributionStr = "";

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
            set 
            { 
                initStateDistribution = value; 
                initStateDistributionStr = initStateDistribution == null ? "None" : NumpyHelper.NumpyParser.ParseArray(initStateDistribution); 
            } 
        }


        private double[,] transitionMatrix = null;
        private string transitionMatrixStr = "";

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
            set
            {
                transitionMatrix = value;
                transitionMatrixStr = transitionMatrix == null ? "None" : NumpyHelper.NumpyParser.ParseArray(transitionMatrix);
            }
        }


        private double[,] observationMatrix = null;
        private string observationMatrixStr = "";

        /// <summary>
        /// The observation matrix.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observations")]
        [Description("The observation matrix.")]
        [Category("ModelState")]
        public double[,] ObservationMatrix
        {
            get => observationMatrix;
            set 
            {
                observationMatrix = value;
                observationMatrixStr = observationMatrix == null ? "None" : NumpyHelper.NumpyParser.ParseArray(observationMatrix);
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelParams"/> class.
        /// </summary>
        public ModelParams()
        {
            NumStates = 2;
            Dimensions = 2;
            ObservationType = ObservationType.Gaussian;
        }

        public IObservable<ModelParams> Process()
        {
            return Observable.Return(
                new ModelParams()  
                { 
                    NumStates = NumStates, 
                    Dimensions = Dimensions, 
                    ObservationType = ObservationType, 
                    InitStateDistribution = InitStateDistribution, 
                    TransitionMatrix = TransitionMatrix,
                    ObservationMatrix = ObservationMatrix
                });
        }

        public IObservable<ModelParams> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, item => 
            {
                return new ModelParams() 
                { 
                    NumStates = NumStates, 
                    Dimensions = Dimensions, 
                    ObservationType = ObservationType, 
                    InitStateDistribution = InitStateDistribution, 
                    TransitionMatrix = TransitionMatrix,
                    ObservationMatrix = ObservationMatrix
                };
            });
        }

        public IObservable<ModelParams> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject => 
            {
                var numStatesPyObj = pyObject.GetAttr<int>("num_states");
                var dimensionsPyObj = pyObject.GetAttr<int>("dimensions");
                var observationTypeStrPyObj = pyObject.GetAttr<string>("observation_type");
                var initStateDistributionPyObj = (double[])pyObject.GetArrayAttr("init_state_distn");
                var transitionMatrixPyObj = (double[,])pyObject.GetArrayAttr("transitions");
                var observationMatrixPyObj = (double[,])pyObject.GetArrayAttr("observations");

                observationTypeStrLookup.TryGetValue(observationTypeStrPyObj, out var observationTypePyObj);

                return new ModelParams() 
                { 
                    NumStates = numStatesPyObj, 
                    Dimensions = dimensionsPyObj, 
                    ObservationType = observationTypePyObj, 
                    InitStateDistribution = initStateDistributionPyObj, 
                    TransitionMatrix = transitionMatrixPyObj,
                    ObservationMatrix = observationMatrixPyObj
                };
            });
        }

        public override string ToString()
        {
            return $"num_states={numStatesStr}, dimensions={dimensionsStr}, observation_type={observationTypeStr}, init_state_distn={initStateDistributionStr}, transitions={transitionMatrixStr}, observations={observationMatrixStr}";
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

