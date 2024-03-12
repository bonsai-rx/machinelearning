using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Newtonsoft.Json;
using Python.Runtime;

namespace Bonsai.ML.LinearDynamicalSystems.LinearRegression
{

    /// <summary>
    /// 2D grid parameters used for calculating the PDF of a multivariate distribution
    /// </summary>
    [Description("2D grid parameters used for calculating the PDF of a multivariate distribution")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class GridParameters
    {

        private double _x0 = 0;
        private double _x1 = 1;
        private int _xsteps = 100;

        private double _y0 = 0;
        private double _y1 = 1;
        private int _ysteps = 100;

        private string _x0String;
        private string _x1String;
        private string _xstepsString;

        private string _y0String;
        private string _y1String;
        private string _ystepsString;

        /// <summary>
        /// Lower bound of the x axis
        /// </summary>
        [JsonProperty("x0")]
        [Description("Lower bound of the x axis")]
        public double X0
        {
            get
            {
                return _x0;
            }
            set
            {
                _x0 = value;
                _x0String = double.IsNaN(_x0) ? "None" : _x0.ToString();

            }
        }
    
        /// <summary>
        /// Upper bound of the x axis
        /// </summary>
        [JsonProperty("x1")]
        [Description("Upper bound of the x axis")]
        public double X1
        {
            get
            {
                return _x1;
            }
            set
            {
                _x1 = value;
                _x1String = double.IsNaN(_x1) ? "None" : _x1.ToString();
            }
        }
        
        /// <summary>
        /// Number of steps along the x axis
        /// </summary>
        [JsonProperty("xsteps")]
        [Description("Number of steps along the x axis")]
        public int Xsteps
        {
            get
            {
                return _xsteps;
            }
            set
            {
                _xsteps = value >= 0 ? value : _xsteps;
                _xstepsString = _xsteps.ToString();
            }
        }

        /// <summary>
        /// Lower bound of the y axis
        /// </summary>
        [JsonProperty("y0")]
        [Description("Lower bound of the y axis")]
        public double Y0
        {
            get
            {
                return _y0;
            }
            set
            {
                _y0 = value;
                _y0String = double.IsNaN(_y0) ? "None" : _y0.ToString();
            }
        }
    
        /// <summary>
        /// Upper bound of the y axis
        /// </summary>
        [JsonProperty("y1")]
        [Description("Upper bound of the y axis")]
        public double Y1
        {
            get
            {
                return _y1;
            }
            set
            {
                _y1 = value;
                _y1String = double.IsNaN(_y1) ? "None" : _y1.ToString();
            }
        }
        
        /// <summary>
        /// Number of steps along the y axis
        /// </summary>
        [JsonProperty("ysteps")]
        [Description("Number of steps along the y axis")]
        public int Ysteps
        {
            get
            {
                return _ysteps;
            }
            set
            {
                _ysteps = value;
                _ystepsString = _ysteps.ToString();
            }
        }

        /// <summary>
        /// Generates grid parameters
        /// </summary>    
        public IObservable<GridParameters> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new GridParameters {
    				X0 = _x0,
                    X1 = _x1,
                    Xsteps = _xsteps,
    				Y0 = _y0,
                    Y1 = _y1,
                    Ysteps = _ysteps,
    			}));
        }

        /// <summary>
        /// Gets the grid parameters from a PyObject of a Kalman Filter Linear Regression Model
        /// </summary>
        public IObservable<GridParameters> Process(IObservable<PyObject> source)
        {
    		return Observable.Select(source, pyObject =>
    		{
                return ConvertPyObject(pyObject);
            });
        }

        public static GridParameters ConvertPyObject(PyObject pyObject)
        {
            var x0PyObj = pyObject.GetAttr<double>("x0");
            var x1PyObj = pyObject.GetAttr<double>("x1");
            var xstepsPyObj = pyObject.GetAttr<int>("xsteps");

            var y0PyObj = pyObject.GetAttr<double>("y0");
            var y1PyObj = pyObject.GetAttr<double>("y1");
            var ystepsPyObj = pyObject.GetAttr<int>("ysteps");

            return new GridParameters {
                X0 = x0PyObj,
                X1 = x1PyObj,
                Xsteps = xstepsPyObj,
                Y0 = y0PyObj,
                Y1 = y1PyObj,
                Ysteps = ystepsPyObj,
            };
        }
    
        /// <summary>
        /// Generates grid parameters on each input
        /// </summary>
        public IObservable<GridParameters> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, x =>
                new GridParameters {
    				X0 = _x0,
                    X1 = _x1,
                    Xsteps = _xsteps,
    				Y0 = _y0,
                    Y1 = _y1,
                    Ysteps = _ysteps,
                });
        }

        /// <inheritdoc/>
        public override string ToString()
        {

            return $"x0={_x0},x1={_x1},xsteps={_xsteps},y0={_y0},y1={_y1},ysteps={_ysteps}";
        }
    }
}
