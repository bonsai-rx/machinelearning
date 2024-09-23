using System.ComponentModel;
using Newtonsoft.Json;
using System;
using System.Reactive.Linq;
using Bonsai.ML.Python;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.LinearDynamicalSystems.LinearRegression
{

    /// <summary>
    /// Represents an operator that creates the model parameters for a Kalman Filter Linear Regression python class
    /// </summary>
    [Combinator]
    [Description("Creates the model parameters used for initializing a Kalman Filter Linear Regression (KFLR) python class")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class KFModelParameters
    {

        private double _likelihood_precision_coef;

        private double _prior_precision_coef;
    
        private int _n_features;

        private double[,] _x = null;
        private double[,] _p = null;
    
        private string _likelihood_precision_coefString;
        private string _prior_precision_coefString;
        private string _n_featuresString;
        private string _xString;
        private string _pString;

        /// <summary>
        /// Gets or sets the likelihood precision coefficient.
        /// </summary>
        [JsonProperty("likelihood_precision_coef")]
        [Description("The likelihood precision coefficient.")]
        [Category("Parameters")]
        public double LikelihoodPrecisionCoefficient
        {
            get
            {
                return _likelihood_precision_coef;
            }
            set
            {
                _likelihood_precision_coef = value;
                _likelihood_precision_coefString = double.IsNaN(_likelihood_precision_coef) ? "None" : _likelihood_precision_coef.ToString();
            }
        }
    
        /// <summary>
        /// Gets or sets the prior precision coefficient.
        /// </summary>
        [JsonProperty("prior_precision_coef")]
        [Description("The prior precision coefficient.")]
        [Category("Parameters")]
        public double PriorPrecisionCoefficient
        {
            get
            {
                return _prior_precision_coef;
            }
            set
            {
                _prior_precision_coef = value;
                _prior_precision_coefString = double.IsNaN(_prior_precision_coef) ? "None" : _prior_precision_coef.ToString();
            }
        }
    
        /// <summary>
        /// Gets or sets the number of features present in the model.
        /// </summary>
        [JsonProperty("n_features")]
        [Description("The number of features.")]
        [Category("Parameters")]
        public int NumFeatures
        {
            get
            {
                return _n_features;
            }
            set
            {
                _n_features = value;
                _n_featuresString = _n_features.ToString();
            }
        }

        /// <summary>
        /// Gets or sets the matrix representing the mean of the state.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("x")]
        [Description("The matrix representing the mean of the state.")]
        [Category("ModelState")]
        public double[,] X
        {
            get
            {
                return _x;
            }
            set
            {
                _x = value;
                _xString = _x == null ? "None" : StringFormatter.FormatToPython(_x);
            }
        }

        /// <summary>
        /// Gets or sets the matrix representing the covariance between state components.
        /// </summary>
        [XmlIgnore]
        [JsonProperty("P")]
        [Description("The matrix representing the covariance between state components.")]
        [Category("ModelState")]
        public double[,] P
        {
            get
            {
                return _p;
            }
            set
            {
                _p = value;
                _pString = _p == null ? "None" : StringFormatter.FormatToPython(_p);
            }
        }

        /// <summary>
        /// Constructs a KF Model Parameters class.
        /// </summary>
        public KFModelParameters ()
        {
            LikelihoodPrecisionCoefficient = 25;
            PriorPrecisionCoefficient = 2;
        }

        /// <summary>
        /// Generates parameters for a Kalman Filter Linear Regression Model
        /// </summary>
        public IObservable<KFModelParameters> Process()
        {
            return Observable.Defer(() => Observable.Return(
                new KFModelParameters {
                    LikelihoodPrecisionCoefficient = _likelihood_precision_coef,
                    PriorPrecisionCoefficient = _prior_precision_coef,
                    NumFeatures = _n_features,
                    X = _x,
                    P = _p
                }));
        }

        /// <summary>
        /// Gets the model parameters from a PyObject of a Kalman Filter Linear Regression Model
        /// </summary>
        public IObservable<KFModelParameters> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var likelihood_precision_coefPyObj = pyObject.GetAttr<double>("likelihood_precision_coef");
                var prior_precision_coefPyObj = pyObject.GetAttr<double>("prior_precision_coef");
                var n_featuresPyObj = pyObject.GetAttr<int>("n_features");

                return new KFModelParameters {
                    LikelihoodPrecisionCoefficient = likelihood_precision_coefPyObj,
                    PriorPrecisionCoefficient = _prior_precision_coef,
                    NumFeatures = n_featuresPyObj,
                    X = _x,
                    P = _p
                };
            });
        }
    
        /// <summary>
        /// Generates parameters for a Kalman Filter Linear Regression Model on each input
        /// </summary>
        public IObservable<KFModelParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, x =>
                new KFModelParameters {
                    LikelihoodPrecisionCoefficient = _likelihood_precision_coef,
                    PriorPrecisionCoefficient = _prior_precision_coef,
                    NumFeatures = _n_features,
                    X = _x,
                    P = _p
                });
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"likelihood_precision_coef={_likelihood_precision_coefString}, prior_precision_coef={_prior_precision_coefString}, n_features={_n_featuresString}, x={_xString}, P={_pString}";
        }
    }

}