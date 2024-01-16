namespace Bonsai.ML.LinearDynamicalSystems
{

    using static PythonHelper;

    /// <summary>
    /// Model parameters for a Kalman Filter python class
    /// </summary>
    [System.ComponentModel.DescriptionAttribute("Model parameters for a Kalman Filter python class")]
    [Bonsai.CombinatorAttribute()]
    [Bonsai.WorkflowElementCategoryAttribute(Bonsai.ElementCategory.Source)]
    public class Model
    {

        private double _pos_x0;
    
        private double _pos_y0;
    
        private double _vel_x0;
    
        private double _vel_y0;
    
        private double _acc_x0;
    
        private double _acc_y0;
    
        private double _sigma_a;
    
        private double _sigma_x;
    
        private double _sigma_y;
    
        private double _sqrt_diag_V0_value;
    
        private int _fps;

        /// <summary>
        /// x position at time 0
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="pos_x0")]
        [System.ComponentModel.DescriptionAttribute("x position at time 0")]
        public double Pos_x0
        {
            get
            {
                return _pos_x0;
            }
            set
            {
                _pos_x0 = value;
            }
        }
    
        /// <summary>
        /// y position at time 0
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="pos_y0")]
        [System.ComponentModel.DescriptionAttribute("y position at time 0")]
        public double Pos_y0
        {
            get
            {
                return _pos_y0;
            }
            set
            {
                _pos_y0 = value;
            }
        }
    
        /// <summary>
        /// x velocity at time 0
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="vel_x0")]
        [System.ComponentModel.DescriptionAttribute("x velocity at time 0")]
        public double Vel_x0
        {
            get
            {
                return _vel_x0;
            }
            set
            {
                _vel_x0 = value;
            }
        }
    
        /// <summary>
        /// y velocity at time 0
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="vel_y0")]
        [System.ComponentModel.DescriptionAttribute("y velocity at time 0")]
        public double Vel_y0
        {
            get
            {
                return _vel_y0;
            }
            set
            {
                _vel_y0 = value;
            }
        }
    
        /// <summary>
        /// x acceleration at time 0
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="acc_x0")]
        [System.ComponentModel.DescriptionAttribute("x acceleration at time 0")]
        public double Acc_x0
        {
            get
            {
                return _acc_x0;
            }
            set
            {
                _acc_x0 = value;
            }
        }
    
        /// <summary>
        /// x velocity at time 0
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="acc_y0")]
        [System.ComponentModel.DescriptionAttribute("x velocity at time 0")]
        public double Acc_y0
        {
            get
            {
                return _acc_y0;
            }
            set
            {
                _acc_y0 = value;
            }
        }
    
        /// <summary>
        /// covariance of a
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="sigma_a")]
        [System.ComponentModel.DescriptionAttribute("covariance of a")]
        public double Sigma_a
        {
            get
            {
                return _sigma_a;
            }
            set
            {
                _sigma_a = value;
            }
        }
    
        /// <summary>
        /// covariance of x
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="sigma_x")]
        [System.ComponentModel.DescriptionAttribute("covariance of x")]
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
        /// covariance of y
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="sigma_y")]
        [System.ComponentModel.DescriptionAttribute("covariance of y")]
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
    
        /// <summary>
        /// v0
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="sqrt_diag_V0_value")]
        [System.ComponentModel.DescriptionAttribute("v0")]
        public double Sqrt_diag_V0_value
        {
            get
            {
                return _sqrt_diag_V0_value;
            }
            set
            {
                _sqrt_diag_V0_value = value;
            }
        }
    
        /// <summary>
        /// frames per second
        /// </summary>
        [YamlDotNet.Serialization.YamlMemberAttribute(Alias="fps")]
        [System.ComponentModel.DescriptionAttribute("frames per second")]
        public int Fps
        {
            get
            {
                return _fps;
            }
            set
            {
                _fps = value;
            }
        }

        public System.IObservable<Model> Process()
        {
    		return System.Reactive.Linq.Observable.Defer(() => System.Reactive.Linq.Observable.Return(
    			new Model {
    				Pos_x0 = _pos_x0,
    				Pos_y0 = _pos_y0,
    				Vel_x0 = _vel_x0,
    				Vel_y0 = _vel_y0,
    				Acc_x0 = _acc_x0,
    				Acc_y0 = _acc_y0,
    				Sigma_a = _sigma_a,
    				Sigma_x = _sigma_x,
    				Sigma_y = _sigma_y,
    				Sqrt_diag_V0_value = _sqrt_diag_V0_value,
    				Fps = _fps
    			}));
        }
    
        public System.IObservable<Model> Process<TSource>(System.IObservable<TSource> source)
        {
    		if (typeof(TSource) == typeof(Python.Runtime.PyObject))
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    			{
    				using(Python.Runtime.Py.GIL())
    				{
    					dynamic input = x;
    					Python.Runtime.PyObject pyObject = (Python.Runtime.PyObject)input;
    					var pos_x0PyObj = GetPythonAttribute<double>(pyObject, "pos_x0");
    					var pos_y0PyObj = GetPythonAttribute<double>(pyObject, "pos_y0");
    					var vel_x0PyObj = GetPythonAttribute<double>(pyObject, "vel_x0");
    					var vel_y0PyObj = GetPythonAttribute<double>(pyObject, "vel_y0");
    					var acc_x0PyObj = GetPythonAttribute<double>(pyObject, "acc_x0");
    					var acc_y0PyObj = GetPythonAttribute<double>(pyObject, "acc_y0");
    					var sigma_aPyObj = GetPythonAttribute<double>(pyObject, "sigma_a");
    					var sigma_xPyObj = GetPythonAttribute<double>(pyObject, "sigma_x");
    					var sigma_yPyObj = GetPythonAttribute<double>(pyObject, "sigma_y");
    					var sqrt_diag_V0_valuePyObj = GetPythonAttribute<double>(pyObject, "sqrt_diag_V0_value");
    					var fpsPyObj = GetPythonAttribute<int>(pyObject, "fps");
					
    					return new Model {
    						Pos_x0 = pos_x0PyObj,
    						Pos_y0 = pos_y0PyObj,
    						Vel_x0 = vel_x0PyObj,
    						Vel_y0 = vel_y0PyObj,
    						Acc_x0 = acc_x0PyObj,
    						Acc_y0 = acc_y0PyObj,
    						Sigma_a = sigma_aPyObj,
    						Sigma_x = sigma_xPyObj,
    						Sigma_y = sigma_yPyObj,
    						Sqrt_diag_V0_value = sqrt_diag_V0_valuePyObj,
    						Fps = fpsPyObj
    					};
    				}
    			});
    		}
    		else
    		{
    			return System.Reactive.Linq.Observable.Select(source, x =>
    				new Model {
    					Pos_x0 = _pos_x0,
    					Pos_y0 = _pos_y0,
    					Vel_x0 = _vel_x0,
    					Vel_y0 = _vel_y0,
    					Acc_x0 = _acc_x0,
    					Acc_y0 = _acc_y0,
    					Sigma_a = _sigma_a,
    					Sigma_x = _sigma_x,
    					Sigma_y = _sigma_y,
    					Sqrt_diag_V0_value = _sqrt_diag_V0_value,
    					Fps = _fps
    				});
    		}
        }
    
        public override string ToString()
        {
    		string output = "";
    		foreach (var prop in typeof(Model).GetProperties())
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