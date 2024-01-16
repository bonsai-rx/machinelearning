namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

    /// <summary>
    /// Prediction of a Kalman Filter python class (state vector and covariance matrix)
    /// </summary>
    [System.ComponentModel.DescriptionAttribute("Prediction of a Kalman Filter python class (predicted position and measurement of" +
        " uncertainty)")]    
    [Bonsai.CombinatorAttribute()]
    [Bonsai.WorkflowElementCategoryAttribute(Bonsai.ElementCategory.Source)]
    public class Prediction
    {
    
        private double _x;
    
        private double _y;
    
        private double _sigma_x;
    
        private double _sigma_y;
    
        /// <summary>
        /// predicted x coordinate
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="x")]
        [System.ComponentModel.DescriptionAttribute("predicted x coordinate")]
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
        /// predicted y coordinate
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="y")]
        [System.ComponentModel.DescriptionAttribute("predicted y coordinate")]
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
    
        /// <summary>
        /// confidence in predicted x coordinate
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="sigma_x")]
        [System.ComponentModel.DescriptionAttribute("confidence in predicted x coordinate")]
        public double Sigma_x
        {
            get
            {
                return _sigma_x;
            }
            set
            {
                _sigma_x = value;
            }
        }
    
        /// <summary>
        /// confidence in predicted y coordinate
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="sigma_y")]
        [System.ComponentModel.DescriptionAttribute("confidence in predicted y coordinate")]
        public double Sigma_y
        {
            get
            {
                return _sigma_y;
            }
            set
            {
                _sigma_y = value;
            }
        }
    
        public System.IObservable<Prediction> Process()
        {
    		return System.Reactive.Linq.Observable.Defer(() => System.Reactive.Linq.Observable.Return(
    			new Prediction {
    				X = _x,
    				Y = _y,
    				Sigma_x = _sigma_x,
    				Sigma_y = _sigma_y
    			}));
        }
    
        public System.IObservable<Prediction> Process<TSource>(System.IObservable<TSource> source)
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
    					var sigma_xPyObj = GetPythonAttribute<double>(pyObject, "sigma_x");
    					var sigma_yPyObj = GetPythonAttribute<double>(pyObject, "sigma_y");
					
    					return new Prediction {
    						X = xPyObj,
    						Y = yPyObj,
    						Sigma_x = sigma_xPyObj,
    						Sigma_y = sigma_yPyObj
    					};
    				}
    			});
    		}
    		else
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    				new Prediction {
    					X = _x,
    					Y = _y,
    					Sigma_x = _sigma_x,
    					Sigma_y = _sigma_y
    				});
    		}
        }
    
        public override string ToString()
        {
    		string output = "";
    		foreach (var prop in typeof(Prediction).GetProperties())
    		{
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
    			return output.TrimEnd(',');
    		}
    		catch
    		{
    			return "";
    		}

        }
    }
}