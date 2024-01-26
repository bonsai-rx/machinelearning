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
    /// Model parameters for a Kalman Filter Kinematics python class
    /// </summary>
    [Description("Model parameters for a Kalman Filter Kinematics (KFK) model")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class KalmanFilterKinematicsModel
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
        [YamlMember(Alias="pos_x0")]
        [Description("x position at time 0")]
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
        [YamlMember(Alias="pos_y0")]
        [Description("y position at time 0")]
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
        [YamlMember(Alias="vel_x0")]
        [Description("x velocity at time 0")]
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
        [YamlMember(Alias="vel_y0")]
        [Description("y velocity at time 0")]
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
        [YamlMember(Alias="acc_x0")]
        [Description("x acceleration at time 0")]
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
        [YamlMember(Alias="acc_y0")]
        [Description("x velocity at time 0")]
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
        [YamlMember(Alias="sigma_a")]
        [Description("covariance of a")]
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
        [YamlMember(Alias="sigma_x")]
        [Description("covariance of x")]
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
        [YamlMember(Alias="sigma_y")]
        [Description("covariance of y")]
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
        [YamlMember(Alias="sqrt_diag_V0_value")]
        [Description("v0")]
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
        [YamlMember(Alias="fps")]
        [Description("frames per second")]
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

        public IObservable<KalmanFilterKinematicsModel> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new KalmanFilterKinematicsModel {
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
    
        public IObservable<KalmanFilterKinematicsModel> Process<TSource>(IObservable<TSource> source)
        {
    		if (typeof(TSource) == typeof(PyObject))
    		{
    			return Observable.Select(source, x =>
    			{
    				using(Py.GIL())
    				{
    					dynamic input = x;
    					PyObject pyObject = (PyObject)input;
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
					
    					return new KalmanFilterKinematicsModel {
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
    			return Observable.Select(source, x =>
    				new KalmanFilterKinematicsModel {
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
    		foreach (var prop in typeof(KalmanFilterKinematicsModel).GetProperties())
    		{
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
    			return output.TrimEnd(',');
    		}
    		catch
    		{
    			return "";
    		}

        }
    }

}