using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

    /// <summary>
    /// State of a Kalman Filter (mean vector and covariance matrix)
    /// </summary>
    [Description("State of a Kalman Filter (mean vector and covariance matrix)")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class State
    {

        private List<List<double>> _x = new List<List<double>>();
    
        private List<List<double>> _p = new List<List<double>>();


        /// <summary>
        /// Mean vector - n x 1 dimensional matrix where n is number of features
        /// </summary>
        [XmlIgnore()]
        [YamlMember(Alias="x")]
        [Description("Mean vector")]
        public List<List<double>> X
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
        public List<List<double>> P
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
                using (Py.GIL())
                {
                    var xPyList = (List<object>)GetPythonAttribute(pyObject, "x");
                    var xPyObj = Enumerable.ToList(Enumerable.Select(Enumerable.Cast<List<object>>(xPyList), subList0 => Enumerable.ToList(Enumerable.OfType<double>(subList0))));
                    var PPyList = (List<object>)GetPythonAttribute(pyObject, "P");
                    var PPyObj = Enumerable.ToList(Enumerable.Select(Enumerable.Cast<List<object>>(PPyList), subList0 => Enumerable.ToList(Enumerable.OfType<double>(subList0))));
                
                    return new State {
                        X = xPyObj,
                        P = PPyObj
                    };
                }
            });
        }
    }
}