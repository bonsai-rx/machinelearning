using System.ComponentModel;
using System;
using System.Xml.Serialization;
using System.Reactive.Linq;
using Newtonsoft.Json;

namespace Bonsai.ML.LinearDynamicalSystems
{
    /// <summary>
    /// State component of a Kalman Filter
    /// </summary>
    [Description("State component of a Kalman Filter")]  
    public class StateComponent
    {

        private double _mean;

        private double _variance;

        /// <summary>
        /// mean
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("mean")]
        [Description("mean")]
        public double Mean
        {
            get
            {
                return _mean;
            }
            set
            {
                _mean = value;
            }
        }

        /// <summary>
        /// variance
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("variance")]
        [Description("variance")]
        public double Variance
        {
            get
            {
                return _variance;
            }
            set
            {
                _variance = value;
            }
        }

        /// <summary>
        /// Base constructor of a state component
        /// </summary>
        public StateComponent() 
        {
        }

        /// <summary>
        /// Constructs a state compenent from the full state and covariance matrices given an index
        /// </summary>
        public StateComponent(double[,] X, double[,] P, int i) 
        {
            Mean = X[i,0];
            Variance = Sigma(P[i,i]);
        }

        private double Sigma(double variance)
        {
            return 2 * Math.Sqrt(variance);
        }

        /// <summary>
        /// Generates a state component
        /// </summary>
        public IObservable<StateComponent> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new StateComponent {
                    Mean = _mean,
                    Variance = _variance
    			}));
        }

        /// <summary>
        /// Generates a state component
        /// </summary>
        public IObservable<StateComponent> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, pyObject =>
            {
                return new StateComponent {
                    Mean = _mean,
                    Variance = _variance
                };
            });
        }
    }
}
