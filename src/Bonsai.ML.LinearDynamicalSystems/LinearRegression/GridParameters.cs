using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Newtonsoft.Json;
using Python.Runtime;
using Bonsai.ML.Python;

namespace Bonsai.ML.LinearDynamicalSystems.LinearRegression
{

    /// <summary>
    /// Represents an operator that creates the 2D grid parameters used for calculating the PDF of a multivariate distribution.
    /// </summary>
    [Combinator]
    [Description("Creates the 2D grid parameters used for calculating the PDF of a multivariate distribution.")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class GridParameters
    {

        private double _x0 = 0;
        private double _x1 = 1;
        private int _xsteps = 100;

        private double _y0 = 0;
        private double _y1 = 1;
        private int _ysteps = 100;

        /// <summary>
        /// Gets or sets the lower bound of the X axis.
        /// </summary>
        [JsonProperty("x0")]
        [Description("The lower bound of the X axis.")]
        public double X0
        {
            get
            {
                return _x0;
            }
            set
            {
                _x0 = value;
            }
        }
    
        /// <summary>
        /// Gets or sets the upper bound of the X axis.
        /// </summary>
        [JsonProperty("x1")]
        [Description("The upper bound of the X axis.")]
        public double X1
        {
            get
            {
                return _x1;
            }
            set
            {
                _x1 = value;
            }
        }
        
        /// <summary>
        /// Gets or sets the number of steps along the X axis.
        /// </summary>
        [JsonProperty("xsteps")]
        [Description("The number of steps along the X axis.")]
        public int XSteps
        {
            get
            {
                return _xsteps;
            }
            set
            {
                _xsteps = value >= 0 ? value : _xsteps;
            }
        }

        /// <summary>
        /// Gets or sets the lower bound of the Y axis.
        /// </summary>
        [JsonProperty("y0")]
        [Description("The lower bound of the Y axis.")]
        public double Y0
        {
            get
            {
                return _y0;
            }
            set
            {
                _y0 = value;
            }
        }
    
        /// <summary>
        /// Gets or sets the upper bound of the Y axis.
        /// </summary>
        [JsonProperty("y1")]
        [Description("The upper bound of the Y axis.")]
        public double Y1
        {
            get
            {
                return _y1;
            }
            set
            {
                _y1 = value;
            }
        }
        
        /// <summary>
        /// Gets or sets the number of steps along the Y axis.
        /// </summary>
        [JsonProperty("ysteps")]
        [Description("The number of steps along the Y axis.")]
        public int YSteps
        {
            get
            {
                return _ysteps;
            }
            set
            {
                _ysteps = value;
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
                    XSteps = _xsteps,
                    Y0 = _y0,
                    Y1 = _y1,
                    YSteps = _ysteps,
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

        /// <summary>
        /// Converts a PyObject, represeting a Kalman Filter Linear Regression Model, into a GridParameters object
        /// </summary>
        public static GridParameters ConvertPyObject(PyObject pyObject)
        {
            var x0PyObj = pyObject.GetAttr<double>("x0");
            var x1PyObj = pyObject.GetAttr<double>("x1");
            var xStepsPyObj = pyObject.GetAttr<int>("xsteps");

            var y0PyObj = pyObject.GetAttr<double>("y0");
            var y1PyObj = pyObject.GetAttr<double>("y1");
            var yStepsPyObj = pyObject.GetAttr<int>("ysteps");

            return new GridParameters {
                X0 = x0PyObj,
                X1 = x1PyObj,
                XSteps = xStepsPyObj,
                Y0 = y0PyObj,
                Y1 = y1PyObj,
                YSteps = yStepsPyObj,
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
                    XSteps = _xsteps,
                    Y0 = _y0,
                    Y1 = _y1,
                    YSteps = _ysteps,
                });
        }

        /// <inheritdoc/>
        public override string ToString()
        {

            return $"x0={_x0}, x1={_x1}, xsteps={_xsteps}, y0={_y0}, y1={_y1}, ysteps={_ysteps}";
        }
    }
}
