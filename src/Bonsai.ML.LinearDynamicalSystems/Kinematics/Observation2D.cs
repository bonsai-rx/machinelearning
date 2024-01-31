using System.ComponentModel;
using YamlDotNet.Serialization;
using System;
using System.Reactive.Linq;
using Python.Runtime;

namespace Bonsai.ML.LinearDynamicalSystems.Kinematics
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
        private string xString = "";
        private string yString = "";

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
                xString = double.IsNaN(_x) ? "None" : _x.ToString();
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
                yString = double.IsNaN(_y) ? "None" : _y.ToString();
            }
        }

        /// <summary>
        /// Generates a 2D observation
        /// </summary>    
        public IObservable<Observation2D> Process()
        {
    		return Observable.Defer(() => Observable.Return(
    			new Observation2D {
    				X = _x,
    				Y = _y
    			}));
        }
    
        /// <summary>
        /// Generates a 2D observation on each input
        /// </summary>
        public IObservable<Observation2D> Process<TSource>(IObservable<TSource> source)
        {
            return Observable.Select(source, x =>
                new Observation2D {
                    X = _x,
                    Y = _y
                });
        }

        public override string ToString()
        {

            return $"x={xString},y={yString}";
        }
    }

}