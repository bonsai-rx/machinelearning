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
    /// Prediction of a Kalman Filter python class (state vector and covariance matrix)
    /// </summary>
    [Description("State of a Kalman Filter python class (mean and covariance matrices)")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class State
    {

        private List<List<double>> _x = new List<List<double>>();
    
        private List<List<double>> _p = new List<List<double>>();


        /// <summary>
        /// State vector
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
        /// Covariance matrix
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

        public override string ToString()
        {
            string output = "";
            foreach (var prop in typeof(State).GetProperties())
            {
                // Get the YamlMemberAttribute of the property
                var yamlAttr = CustomAttributeExtensions.GetCustomAttribute<YamlMemberAttribute>(prop);
                var yamlAlias = yamlAttr != null && !string.IsNullOrWhiteSpace(yamlAttr.Alias) ? yamlAttr.Alias : char.ToLower(prop.Name[0]) + prop.Name.Substring(1);
                var value = prop.GetValue(this, null);
                if (value is double && double.IsNaN((double)value))
                {
                    output += yamlAlias + "=None,";
                }
                else
                {
                    output += yamlAlias + "=" + value + ",";
                }
            }
            try
            {
                return output.TrimEnd(','); // Remove the trailing comma
            }
            catch
            {
                return "";
            }
        }
    }

}