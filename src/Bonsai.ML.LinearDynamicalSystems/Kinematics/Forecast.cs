using System;
using System.Reactive;
using System.Reactive.Linq;
using System.Collections.ObjectModel;
using System.Xml.Serialization;
using System.ComponentModel;
using Newtonsoft.Json;
using Python.Runtime;
using System.Collections.Generic;

namespace Bonsai.ML.LinearDynamicalSystems.Kinematics
{
    /// <summary>
    /// Represents an operator for converting forecasts from a Kalman Filter Kinematics python class into a list of forecasted results.
    /// </summary>
    [Combinator]
    [Description("Forecasts for a Kalman Filter Kinematics (KFK) model.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class Forecast
    {
        /// <summary>
        /// Gets or sets the list of forecast results.
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("forecasts")]
        [Description("The list of forecast results.")]
        public List<ForecastResult> ForecastResults { get; private set; }

        /// <summary>
        /// Converts a PyObject representing a Kalman Filter forecast into a Forecast class representing a list of forecasted results.
        /// </summary>
        public IObservable<Forecast> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject => {

                dynamic pyObj = pyObject;

                var xs = (PyObject[])pyObj[0];
                var Ps = (PyObject[])pyObj[1];
                var dts = (double[])pyObj[2];

                var results = new List<ForecastResult>();

                for (int i = 0; i < xs.Length; i++)
                {
                    double[,] x = (double[,])PythonHelper.ConvertPythonObjectToCSharp(xs[i]);
                    double[,] P = (double[,])PythonHelper.ConvertPythonObjectToCSharp(Ps[i]);
                    var state = new State {X=x, P=P};
                    var kinematicState = new KinematicState(state);

                    var dt = dts[i];
                    var timestep = TimeSpan.FromSeconds(dt);

                    results.Add(new ForecastResult(kinematicState, timestep));
                }

                return new Forecast {
                    ForecastResults = results
                };
            });
        }
    }
}

