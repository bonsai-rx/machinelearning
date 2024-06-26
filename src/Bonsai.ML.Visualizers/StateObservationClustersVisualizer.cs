using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.HiddenMarkovModels;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.WindowsForms;
using OxyPlot.Annotations;
using MathNet.Numerics.LinearAlgebra;
using System.Drawing;
using Bonsai.Dsp;
using Bonsai.Reactive;

[assembly: TypeVisualizer(typeof(StateObservationClustersVisualizer), Target = typeof(StateObservationClusters))]

namespace Bonsai.ML.Visualizers
{
    public class StateObservationClustersVisualizer : DialogTypeVisualizer
    {
        private PlotView view;
        private PlotModel model;
        private List<LineSeries> allLineSeries = null;
        private List<ScatterSeries> allScatterSeries = null;
        private List<OxyColor> colorList = null;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            view = new PlotView
            {
                Dock = DockStyle.Fill,
            };

            model = new PlotModel();

            var xAxis = new LinearAxis
            {
                Position = AxisPosition.Bottom,
                Title = "Dimension0",
            };

            var yAxis = new LinearAxis
            {
                Position = AxisPosition.Left,
                Title = "Dimension1",
            };

            model.Axes.Add(xAxis);
            model.Axes.Add(yAxis);

            view.Model = model;

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));

            if (visualizerService != null)
            {
                visualizerService.AddControl(view);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (value is StateObservationClusters stateObservationClusters)
            {
                var statesCount = stateObservationClusters.Means.GetLength(0);
                if (colorList == null)
                {
                    colorList = new List<OxyColor>();
                    for (int i = 0; i < statesCount; i++)
                    {
                        OxyColor color = OxyPalettes.Jet(statesCount).Colors[i];
                        colorList.Add(color);
                    }
                }

                if (allScatterSeries == null)
                {
                    allScatterSeries = new List<ScatterSeries>();
                    for (int i = 0; i < statesCount; i++)
                    {
                        var scatterSeries = new ScatterSeries
                        {
                            MarkerType = MarkerType.Circle,
                            MarkerSize = 4,
                            MarkerFill = colorList[i],
                            MarkerStroke = OxyColors.Black,
                            MarkerStrokeThickness = 1
                        };
                        allScatterSeries.Add(scatterSeries);
                        model.Series.Add(scatterSeries);
                    }
                }

                if (allLineSeries != null)
                {
                    foreach (var lineSeries in allLineSeries)
                    {
                        lineSeries.Points.Clear();
                    }
                }
                else
                {
                    allLineSeries = new List<LineSeries>();
                    for (int i = 0; i < statesCount; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            var lineSeries = new LineSeries { Color = colorList[i] };
                            allLineSeries.Add(lineSeries);
                            model.Series.Add(lineSeries);
                        }
                    }
                }

                for (int i = 0; i < stateObservationClusters.BatchObservations.GetLength(0); i++)
                {
                    var dim1 = stateObservationClusters.BatchObservations[i, 0];
                    var dim2 = stateObservationClusters.BatchObservations[i, 1];
                    var state = stateObservationClusters.InferredMostProbableStates[i];
                    allScatterSeries[(int)state].Points.Add(new ScatterPoint(dim1, dim2, value: state, tag: state));
                }

                for (int i = 0; i < statesCount; i++)
                {
                    var xMean = stateObservationClusters.Means[i, 0];
                    var yMean = stateObservationClusters.Means[i, 1];

                    var xVar = stateObservationClusters.CovarianceMatrices[i, 0, 0];
                    var yVar = stateObservationClusters.CovarianceMatrices[i, 1, 1];
                    var xyCov = stateObservationClusters.CovarianceMatrices[i, 1, 0];

                    var covariance = Matrix<double>.Build.DenseOfArray(new double[,] {
                        {
                            xVar,
                            xyCov
                        },
                        {
                            xyCov,
                            yVar
                        },
                    });

                    var evd = covariance.Evd();
                    var evals = evd.EigenValues.Real();
                    evals = evals.PointwiseAbsoluteMaximum(0);
                    var evecs = evd.EigenVectors;

                    double angle = Math.Atan2(evecs[1, 0], evecs[0, 0]) * 180 / Math.PI;

                    for (int j = 1; j < 4; j++)
                    {

                        var majorAxis = j * Math.Sqrt(evals[0]);
                        var minorAxis = j * Math.Sqrt(evals[1]);

                        var points = new List<DataPoint>();
                        int numPoints = 100;
                        for (int k = 0; k < numPoints + 1; k++)
                        {
                            double theta = 2 * Math.PI * k / numPoints;
                            double x = majorAxis * Math.Cos(theta);
                            double y = minorAxis * Math.Sin(theta);

                            double xRot = x * Math.Cos(angle) - y * Math.Sin(angle);
                            double yRot = x * Math.Sin(angle) + y * Math.Cos(angle);

                            points.Add(new DataPoint(xMean + xRot, yMean + yRot));
                        }

                        allLineSeries[i * 3 + j - 1].Points.AddRange(points);

                    }
                }
                model.InvalidatePlot(true);
            }
        }

        /// <inheritdoc/>
        public override void Unload()
        {
            allLineSeries = null;
            colorList = null;
            allScatterSeries = null;
            if (!view.IsDisposed)
            {
                view.Dispose();
            }
        }
    }
}
