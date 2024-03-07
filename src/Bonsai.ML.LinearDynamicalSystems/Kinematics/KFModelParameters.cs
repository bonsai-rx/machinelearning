using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using Newtonsoft.Json;

namespace Bonsai.ML.LinearDynamicalSystems.Kinematics
{

    /// <summary>
    /// Model parameters for a Kalman Filter Kinematics python class
    /// </summary>
    [Description("Model parameters for a Kalman Filter Kinematics (KFK) model")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class KFModelParameters
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
        private string pos_x0String;
        private string pos_y0String;
        private string vel_x0String;
        private string vel_y0String;
        private string acc_x0String;
        private string acc_y0String;
        private string sigma_aString;
        private string sigma_xString;
        private string sigma_yString;
        private string sqrt_diag_V0_valueString;
        private string fpsString;

        /// <summary>
        /// x position at time 0
        /// </summary>
        [JsonProperty("pos_x0")]
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
                pos_x0String = double.IsNaN(_pos_x0) ? "None" : _pos_x0.ToString();
            }
        }
    
        /// <summary>
        /// y position at time 0
        /// </summary>
        [JsonProperty("pos_y0")]
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
                pos_y0String = double.IsNaN(_pos_y0) ? "None" : _pos_y0.ToString();
            }
        }
    
        /// <summary>
        /// x velocity at time 0
        /// </summary>
        [JsonProperty("vel_x0")]
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
                vel_x0String = double.IsNaN(_vel_x0) ? "None" : _vel_x0.ToString();
            }
        }
    
        /// <summary>
        /// y velocity at time 0
        /// </summary>
        [JsonProperty("vel_y0")]
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
                vel_y0String = double.IsNaN(_vel_y0) ? "None" : _vel_y0.ToString();
            }
        }
    
        /// <summary>
        /// x acceleration at time 0
        /// </summary>
        [JsonProperty("acc_x0")]
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
                acc_x0String = double.IsNaN(_acc_x0) ? "None" : _acc_x0.ToString();
            }
        }
    
        /// <summary>
        /// x velocity at time 0
        /// </summary>
        [JsonProperty("acc_y0")]
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
                acc_y0String = double.IsNaN(_acc_y0) ? "None" : _acc_y0.ToString();
            }
        }
    
        /// <summary>
        /// covariance of a
        /// </summary>
        [JsonProperty("sigma_a")]
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
                sigma_aString = double.IsNaN(_sigma_a) ? "None" : _sigma_a.ToString();
            }
        }
    
        /// <summary>
        /// covariance of x
        /// </summary>
        [JsonProperty("sigma_x")]
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
                sigma_xString = double.IsNaN(_sigma_x) ? "None" : _sigma_x.ToString();
            }
        }
    
        /// <summary>
        /// covariance of y
        /// </summary>
        [JsonProperty("sigma_y")]
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
                sigma_yString = double.IsNaN(_sigma_y) ? "None" : _sigma_y.ToString();
            }
        }
    
        /// <summary>
        /// v0
        /// </summary>
        [JsonProperty("sqrt_diag_V0_value")]
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
                sqrt_diag_V0_valueString = double.IsNaN(_sqrt_diag_V0_value) ? "None" : _sqrt_diag_V0_value.ToString();
            }
        }
    
        /// <summary>
        /// frames per second
        /// </summary>
        [JsonProperty("fps")]
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
                fpsString = _fps.ToString();
            }
        }

        public KFModelParameters ()
        {
            Sigma_a = 10000;
            Sigma_x = 100;
            Sigma_y = 100;
            Sqrt_diag_V0_value = 0.001;
            Fps = 60;
        }

        /// <summary>
        /// Generates parameters for a Kalman Filter Kinematics Model
        /// </summary>
        public IObservable<KFModelParameters> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new KFModelParameters {
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

        /// <summary>
        /// Gets the model parameters from a PyObject of a Kalman Filter Kinematics Model
        /// </summary>
        public IObservable<KFModelParameters> Process(IObservable<PyObject> source)
        {
    		return Observable.Select(source, pyObject =>
    		{
                var pos_x0PyObj = pyObject.GetAttr<double>("pos_x0");
                var pos_y0PyObj = pyObject.GetAttr<double>("pos_y0");
                var vel_x0PyObj = pyObject.GetAttr<double>("vel_x0");
                var vel_y0PyObj = pyObject.GetAttr<double>("vel_y0");
                var acc_x0PyObj = pyObject.GetAttr<double>("acc_x0");
                var acc_y0PyObj = pyObject.GetAttr<double>("acc_y0");
                var sigma_aPyObj = pyObject.GetAttr<double>("sigma_a");
                var sigma_xPyObj = pyObject.GetAttr<double>("sigma_x");
                var sigma_yPyObj = pyObject.GetAttr<double>("sigma_y");
                var sqrt_diag_V0_valuePyObj = pyObject.GetAttr<double>("sqrt_diag_V0_value");
                var fpsPyObj = pyObject.GetAttr<int>("fps");

                return new KFModelParameters {
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
            });
        }
    
        /// <summary>
        /// Generates parameters for a Kalman Filter Kinematics Model on each input
        /// </summary>
        public IObservable<KFModelParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, x =>
                new KFModelParameters {
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
    
        public override string ToString()
        {

            return $"pos_x0={pos_x0String},pos_y0={pos_y0String},vel_x0={vel_x0String},vel_y0={vel_y0String},acc_x0={acc_x0String},acc_y0={acc_y0String},sigma_a={sigma_aString},sigma_x={sigma_xString},sigma_y={sigma_yString},sqrt_diag_V0_value={sqrt_diag_V0_valueString},fps={fpsString}";
        }
    }

}
