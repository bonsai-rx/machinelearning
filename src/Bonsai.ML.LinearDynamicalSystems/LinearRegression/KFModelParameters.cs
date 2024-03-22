using System.ComponentModel;
using Newtonsoft.Json;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.LinearDynamicalSystems.LinearRegression
{

    /// <summary>
    /// Model parameters for a Kalman Filter Kinematics python class
    /// </summary>
    [Description("Model parameters for a Kalman Filter Linear Regression (KFLR) model")]
    [Combinator()]
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
        /// likelihood precision coefficient
        /// </summary>
        [JsonProperty("likelihood_precision_coef")]
        [Description("likelihood precision coefficient")]
        [Category("Parameters")]
        public double LikelihoodPrecisionCoef
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
        /// prior precision coefficient
        /// </summary>
        [JsonProperty("prior_precision_coef")]
        [Description("prior precision coefficient")]
        [Category("Parameters")]
        public double PriorPrecisionCoef
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
        /// number of features
        /// </summary>
        [JsonProperty("n_features")]
        [Description("number of features")]
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
        /// matrix representing the mean of the state
        /// </summary>
        [XmlIgnore]
        [JsonProperty("x")]
        [Description("matrix representing the mean of the state")]
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
                _xString = _x == null ? "None" : NumpyHelper.NumpyParser.ParseArray(_x);
            }
        }

        /// <summary>
        /// matrix representing the covariance of the state
        /// </summary>
        [XmlIgnore]
        [JsonProperty("P")]
        [Description("matrix representing the covariance of the state")]
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
                _pString = _p == null ? "None" : NumpyHelper.NumpyParser.ParseArray(_p);
            }
        }
    
        public KFModelParameters ()
        {
            LikelihoodPrecisionCoef = 25;
            PriorPrecisionCoef = 2;
        }

        /// <summary>
        /// Generates parameters for a Kalman Filter Linear Regression Model
        /// </summary>
        public IObservable<KFModelParameters> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new KFModelParameters {
    				LikelihoodPrecisionCoef = _likelihood_precision_coef,
                    PriorPrecisionCoef = _prior_precision_coef,
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
                    LikelihoodPrecisionCoef = likelihood_precision_coefPyObj,
                    PriorPrecisionCoef = _prior_precision_coef,
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
    				LikelihoodPrecisionCoef = _likelihood_precision_coef,
                    PriorPrecisionCoef = _prior_precision_coef,
                    NumFeatures = _n_features,
                    X = _x,
                    P = _p
                });
        }
    
        public override string ToString()
        {

            return $"likelihood_precision_coef={_likelihood_precision_coefString},prior_precision_coef={_prior_precision_coefString},n_features={_n_featuresString},x={_xString},P={_pString}";
        }
    }

}