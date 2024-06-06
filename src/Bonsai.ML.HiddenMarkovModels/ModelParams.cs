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
        [JsonProperty("init_state_distribution")]
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
        [JsonProperty("transition_matrix")]
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


        private double[,] observationMeans = null;
        private string observationMeansStr = "";

        /// <summary>
        /// The observation matrix.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observation_matrix")]
        [Description("The observation matrix.")]
        [Category("ModelState")]
        public double[,] ObservationMeans
        {
            get => observationMeans;
            set 
            {
                observationMeans = value;
                observationMeansStr = observationMeans == null ? "None" : NumpyHelper.NumpyParser.ParseArray(observationMeans);
            }
        }


        private double[,,] observationCovs = null;
        private string observationCovsStr = "";

        /// <summary>
        /// The observation matrix.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("observations")]
        [Description("The observation matrix.")]
        [Category("ModelState")]
        public double[,,] ObservationCovs
        {
            get => observationCovs;
            set 
            {
                observationCovs = value;
                observationCovsStr = observationCovs == null ? "None" : NumpyHelper.NumpyParser.ParseArray(observationCovs);
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
                    ObservationMeans = ObservationMeans,
                    ObservationCovs = ObservationCovs
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
                    ObservationMeans = ObservationMeans,
                    ObservationCovs = ObservationCovs
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

                var initStateDistributionPyObj = (double[])pyObject.GetArrayAttr("init_state_distribution");
                var transitionMatrixPyObj = (double[,])pyObject.GetArrayAttr("transition_matrix");
                var observationMeansPyObj = (double[,])pyObject.GetArrayAttr("observation_means");
                var observationCovsPyObj = (double[,,])pyObject.GetArrayAttr("observation_covs");

                observationTypeStrLookup.TryGetValue(observationTypeStrPyObj, out var observationTypePyObj);

                return new ModelParams() 
                { 
                    NumStates = numStatesPyObj, 
                    Dimensions = dimensionsPyObj, 
                    ObservationType = observationTypePyObj, 
                    InitStateDistribution = initStateDistributionPyObj, 
                    TransitionMatrix = transitionMatrixPyObj,
                    ObservationMeans = observationMeansPyObj, 
                    ObservationCovs = observationCovsPyObj
                };
            });
        }

        public override string ToString()
        {
            return $"num_states={numStatesStr}, dimensions={dimensionsStr}, observation_type=\"{observationTypeStr}\", init_state_distribution={initStateDistributionStr}, transition_matrix={transitionMatrixStr}, observation_means={observationMeansStr}, observation_covs={observationCovsStr}";
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

