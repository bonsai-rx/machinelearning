using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Reflection;
using System.Xml.Serialization;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

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
        [YamlMember(Alias="mean")]
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
        [YamlMember(Alias="variance")]
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

        public StateComponent(List<List<double>> X, List<List<double>> P, int i) 
        {
            Mean = X[i][0];
            Variance = Sigma(P[i][i]);
        }

        private double Sigma(double variance)
        {
            return 2 * Math.Sqrt(variance);
        }
    }

    /// <summary>
    /// State of a Kalman Filter (mean vector and covariance matrix)
    /// </summary>
    [Description("State of a Kalman Filter (mean vector and covariance matrix)")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
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
    
        public IObservable<State> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new State {
    				X = _x,
    				P = _p
    			}));
        }
    
        public IObservable<State> Process<TSource>(IObservable<TSource> source)
        {
    		if (typeof(TSource) == typeof(PyObject))
    		{
    			return Observable.Select(source, x =>
    			{
    				using(Py.GIL())
    				{
    					dynamic input = x;
    					PyObject pyObject = (PyObject)input;
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
    		else
    		{
    			return Observable.Select(source, x =>
    				new State {
    					X = _x,
    					P = _p
    				});
    		}
        }
    }
}