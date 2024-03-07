using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.LinearDynamicalSystems
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
        [YamlMember(Alias="x")]
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
        [YamlMember(Alias="P")]
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
                var xPyObj = (double[,])pyObject.GetArrayAttribute("x");
                var PPyObj = (double[,])pyObject.GetArrayAttribute("P");

                return new State {
                    X = xPyObj,
                    P = PPyObj
                };
            });
        }
    }
}
