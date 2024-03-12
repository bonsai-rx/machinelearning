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
    
        private double[,] _mn = null;
    
        private string _likelihood_precision_coefString;
        private string _prior_precision_coefString;
        private string _mnString;

        /// <summary>
        /// likelihood precision coefficient
        /// </summary>
        [JsonProperty("likelihood_precision_coef")]
        [Description("likelihood precision coefficient")]
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
        /// mean of the prior
        /// </summary>
        [XmlIgnore]
        [JsonProperty("mn")]
        [Description("mean of the prior")]
        public double[,] Mn
        {
            get
            {
                return _mn;
            }
            set
            {
                _mn = value;
                if (_mn == null || _mn.Length == 0) _mnString = "None";
                else
                {
                    _mnString = NumpyHelper.NumpyParser.ParseArray(_mn);
                }
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
                    Mn = _mn
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
                var mnPyObj = (double[,])pyObject.GetArrayAttribute("mn");

                return new KFModelParameters {
                    LikelihoodPrecisionCoef = likelihood_precision_coefPyObj,
                    PriorPrecisionCoef = _prior_precision_coef,
                    Mn = mnPyObj
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
                    Mn = _mn
                });
        }
    
        public override string ToString()
        {

            return $"likelihood_precision_coef={_likelihood_precision_coefString},prior_precision_coef={_prior_precision_coefString},mn={_mnString}";
        }
    }

}