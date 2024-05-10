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
    /// Forecasts for a Kalman Filter Kinematics python class
    /// </summary>
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [Description("Forecasts for a Kalman Filter Kinematics (KFK) model")]
    public class Forecast
    {
        private List<ForecastResult> forecastResults;

        /// <summary>
        /// list of forecast results
        /// </summary>
        [XmlIgnore()]
        [JsonProperty("forecasts")]
        [Description("list of forecast results")]
        public List<ForecastResult> ForecastResults
        {
            get
            {
                return forecastResults;
            }
            private set
            {
                forecastResults = value;
            }
        }

        /// <summary>
        /// Grabs the forecasted state of a Kalman Filter model from a type of PyObject
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
                    var kinematicState = new KinematicState().Construct(state);

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

