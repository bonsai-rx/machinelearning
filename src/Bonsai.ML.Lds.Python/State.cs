using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Newtonsoft.Json;
using Bonsai.ML.Python;

namespace Bonsai.ML.Lds.Python
{

    /// <summary>
    /// State of a Kalman Filter (mean vector and covariance matrix)
    /// </summary>
    [Description("State of a Kalman Filter (mean vector and covariance matrix)")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class State
    {

        private double[,] _x;
    
        private double[,] _p;


        /// <summary>
        /// Mean vector - n x 1 dimensional matrix where n is number of features
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("x")]
        [Description("Mean vector")]
        public double[,] X
        {
            get
            {
                return _x;
            }
            set
            {
                _x = value;
            }
        }
    
        /// <summary>
        /// Covariance matrix - n x n dimensional matrix where n is number of features
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("P")]
        [Description("Covariance matrix")]
        public double[,] P
        {
            get
            {
                return _p;
            }
            set
            {
                _p = value;
            }
        }

        /// <summary>
        /// Grabs the state of a Kalman Filter from a type of PyObject
        /// /// </summary>
        public IObservable<State> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var xPyObj = (double[,])pyObject.GetArrayAttr("x");
                var PPyObj = (double[,])pyObject.GetArrayAttr("P");

                return new State {
                    X = xPyObj,
                    P = PPyObj
                };
            });
        }
    }
}
