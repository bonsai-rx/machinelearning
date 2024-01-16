namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

    /// <summary>
    /// Prediction of a Kalman Filter python class (state vector and covariance matrix)
    /// </summary>
    [System.ComponentModel.DescriptionAttribute("State of a Kalman Filter python class (mean and covariance matrices)")]
    [Bonsai.CombinatorAttribute()]
    [Bonsai.WorkflowElementCategoryAttribute(Bonsai.ElementCategory.Source)]
    public class State
    {

        private System.Collections.Generic.List<System.Collections.Generic.List<double>> _x = new System.Collections.Generic.List<System.Collections.Generic.List<double>>();
    
        private System.Collections.Generic.List<System.Collections.Generic.List<double>> _p = new System.Collections.Generic.List<System.Collections.Generic.List<double>>();


        /// <summary>
        /// State vector
        /// </summary>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="x")]
        [System.ComponentModel.DescriptionAttribute("State vector")]
        public System.Collections.Generic.List<System.Collections.Generic.List<double>> X
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
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="P")]
        [System.ComponentModel.DescriptionAttribute("Covariance matrix")]
        public System.Collections.Generic.List<System.Collections.Generic.List<double>> P
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
    
        public System.IObservable<State> Process()
        {
    		return System.Reactive.Linq.Observable.Defer(() => System.Reactive.Linq.Observable.Return(
    			new State {
    				X = _x,
    				P = _p
    			}));
        }
    
        public System.IObservable<State> Process<TSource>(System.IObservable<TSource> source)
        {
    		if (typeof(TSource) == typeof(Python.Runtime.PyObject))
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    			{
    				using(Python.Runtime.Py.GIL())
    				{
    					dynamic input = x;
    					Python.Runtime.PyObject pyObject = (Python.Runtime.PyObject)input;
    					var xPyList = (System.Collections.Generic.List<object>)GetPythonAttribute(pyObject, "x");
    					var xPyObj = System.Linq.Enumerable.ToList(System.Linq.Enumerable.Select(System.Linq.Enumerable.Cast<System.Collections.Generic.List<object>>(xPyList), subList0 => System.Linq.Enumerable.ToList(System.Linq.Enumerable.OfType<double>(subList0))));
    					var PPyList = (System.Collections.Generic.List<object>)GetPythonAttribute(pyObject, "P");
    					var PPyObj = System.Linq.Enumerable.ToList(System.Linq.Enumerable.Select(System.Linq.Enumerable.Cast<System.Collections.Generic.List<object>>(PPyList), subList0 => System.Linq.Enumerable.ToList(System.Linq.Enumerable.OfType<double>(subList0))));
					
    					return new State {
    						X = xPyObj,
    						P = PPyObj
    					};
    				}
    			});
    		}
    		else
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    				new State {
    					X = _x,
    					P = _p
    				});
    		}
        }

        public override string ToString()
        {
            string output = "";
            foreach (var prop in typeof(Prediction).GetProperties())
            {
                // Get the YamlMemberAttribute of the property
                var yamlAttr = System.Reflection.CustomAttributeExtensions.GetCustomAttribute<YamlDotNet.Serialization.YamlMemberAttribute>(prop);
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