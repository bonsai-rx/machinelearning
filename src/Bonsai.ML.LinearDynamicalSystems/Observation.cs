namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

    /// <summary>
    /// Observation of data used by Kalman Filter python class (point(x, y))
    /// </summary>
    [System.ComponentModel.DescriptionAttribute("Observation of data used by Kalman Filter python class (point(x, y))")]
    [Bonsai.CombinatorAttribute()]
    [Bonsai.WorkflowElementCategoryAttribute(Bonsai.ElementCategory.Source)]
    public class Observation
    {

        private double _x;
    
        private double _y;

        /// <summary>
        /// x coordinate
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="x")]
        [System.ComponentModel.DescriptionAttribute("x coordinate")]
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
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="y")]
        [System.ComponentModel.DescriptionAttribute("y coordinate")]
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
    
        public System.IObservable<Observation> Process()
        {
    		return System.Reactive.Linq.Observable.Defer(() => System.Reactive.Linq.Observable.Return(
    			new Observation {
    				X = _x,
    				Y = _y
    			}));
        }
    
        public System.IObservable<Observation> Process<TSource>(System.IObservable<TSource> source)
        {
    		if (typeof(TSource) == typeof(Python.Runtime.PyObject))
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    			{
    				using(Python.Runtime.Py.GIL())
    				{
    					dynamic input = x;
    					Python.Runtime.PyObject pyObject = (Python.Runtime.PyObject)input;
    					var xPyObj = GetPythonAttribute<double>(pyObject, "x");
    					var yPyObj = GetPythonAttribute<double>(pyObject, "y");
					
    					return new Observation {
    						X = xPyObj,
    						Y = yPyObj
    					};
    				}
    			});
    		}
    		else
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    				new Observation {
    					X = _x,
    					Y = _y
    				});
    		}
        }

        public override string ToString()
        {
            string output = "";
            foreach (var prop in typeof(Observation).GetProperties())
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