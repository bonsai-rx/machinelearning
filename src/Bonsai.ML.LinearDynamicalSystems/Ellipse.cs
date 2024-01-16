namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

    /// <summary>
    /// Ellipse parameters for confidence metric
    /// </summary>
    [System.ComponentModel.DescriptionAttribute("Ellipse parameters for confidence metric")]
    [Bonsai.CombinatorAttribute()]
    [Bonsai.WorkflowElementCategoryAttribute(Bonsai.ElementCategory.Source)]
    public class Ellipse
    {

        private double _el_a;

        private double _el_w;

        private double _el_h;

        /// <summary>
        /// ellipse angle
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="el_a")]
        [System.ComponentModel.DescriptionAttribute("ellipse angle")]
        public double El_a
        {
            get
            {
                return _el_a;
            }
            set
            {
                _el_a = value;
            }
        }

        /// <summary>
        /// ellipse width
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="el_w")]
        [System.ComponentModel.DescriptionAttribute("ellipse width")]
        public double El_w
        {
            get
            {
                return _el_w;
            }
            set
            {
                _el_w = value;
            }
        }

        /// <summary>
        /// ellipse height
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="el_h")]
        [System.ComponentModel.DescriptionAttribute("ellipse height")]
        public double El_h
        {
            get
            {
                return _el_h;
            }
            set
            {
                _el_h = value;
            }
        }

        public System.IObservable<Ellipse> Process()
        {
    		return System.Reactive.Linq.Observable.Defer(() => System.Reactive.Linq.Observable.Return(
    			new Ellipse {
    				El_a = _el_a,
    				El_w = _el_w,
    				El_h = _el_h
    			}));
        }
    
        public System.IObservable<Ellipse> Process<TSource>(System.IObservable<TSource> source)
        {
    		if (typeof(TSource) == typeof(Python.Runtime.PyObject))
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    			{
    				using(Python.Runtime.Py.GIL())
    				{
    					dynamic input = x;
    					Python.Runtime.PyObject pyObject = (Python.Runtime.PyObject)input;
    					var el_aPyObj = GetPythonAttribute<double>(pyObject, "el_a");
    					var el_wPyObj = GetPythonAttribute<double>(pyObject, "el_w");
    					var el_hPyObj = GetPythonAttribute<double>(pyObject, "el_h");
					
    					return new Ellipse {
    						El_a = el_aPyObj,
    						El_w = el_wPyObj,
    						El_h = el_hPyObj
    					};
    				}
    			});
    		}
    		else
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    				new Ellipse {
    					El_a = _el_a,
    					El_w = _el_w,
    					El_h = _el_h
    				});
    		}
        }
    
        public override string ToString()
        {
    		string output = "";
    		foreach (var prop in typeof(Ellipse).GetProperties())
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