using Bonsai;
using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Reflection;

namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

    /// <summary>
    /// Observation2D of data used by Kalman Filter python class (point(x, y))
    /// </summary>
    [Description("Observation2D of data used by a Kalman Filter model (point(x, y))")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class Observation2D
    {

        private double _x;
    
        private double _y;

        /// <summary>
        /// x coordinate
        /// </summary>
        [YamlMember(Alias="x")]
        [Description("x coordinate")]
        public double X
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
        /// y coordinate
        /// </summary>
        [YamlMember(Alias="y")]
        [Description("y coordinate")]
        public double Y
        {
            get
            {
                return _y;
            }
            set
            {
                _y = value;
            }
        }
    
        public IObservable<Observation2D> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new Observation2D {
    				X = _x,
    				Y = _y
    			}));
        }
    
        public IObservable<Observation2D> Process<TSource>(IObservable<TSource> source)
        {
    		if (typeof(TSource) == typeof(PyObject))
    		{
    			return Observable.Select(source, x =>
    			{
    				using(Py.GIL())
    				{
    					dynamic input = x;
    					PyObject pyObject = (PyObject)input;
    					var xPyObj = GetPythonAttribute<double>(pyObject, "x");
    					var yPyObj = GetPythonAttribute<double>(pyObject, "y");
					
    					return new Observation2D {
    						X = xPyObj,
    						Y = yPyObj
    					};
    				}
    			});
    		}
    		else
    		{
    			return Observable.Select(source, x =>
    				new Observation2D {
    					X = _x,
    					Y = _y
    				});
    		}
        }

        public override string ToString()
        {
            string output = "";
            foreach (var prop in typeof(Observation2D).GetProperties())
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