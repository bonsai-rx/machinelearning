using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Visualizers;
using Bonsai.ML.HiddenMarkovModels.Observations;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.WindowsForms;
using MathNet.Numerics.LinearAlgebra;

[assembly: TypeVisualizer(typeof(GaussianObservationsClustersVisualizer), Target = typeof(GaussianObservationsStatistics))]

namespace Bonsai.ML.Visualizers
{
    public class GaussianObservationsClustersVisualizer : DialogTypeVisualizer
    {
        private PlotView view;
        private PlotModel model;
        private List<LineSeries> allLineSeries = null;
        private List<ScatterSeries> allScatterSeries = null;
        private List<OxyColor> colorList = null;

        private StatusStrip statusStrip;
        public StatusStrip StatusStrip => statusStrip;

        private int dimension1SelectedIndex = 0;
        public int Dimension1SelectedIndex { get => dimension1SelectedIndex; set => dimension1SelectedIndex = value; }
        private ToolStripComboBox dimension1ComboBox;
        private ToolStripLabel dimension1Label;
        public ToolStripComboBox Dimension1ComboBox => dimension1ComboBox;

        private int dimension2SelectedIndex = 1;
        public int Dimension2SelectedIndex { get => dimension2SelectedIndex; set => dimension2SelectedIndex = value; }
        private ToolStripComboBox dimension2ComboBox;
        private ToolStripLabel dimension2Label;
        public ToolStripComboBox Dimension2ComboBox => dimension2ComboBox;

        public bool BufferData { get; set; } = true;
        public int BufferCount { get; set; } = 250;

        private LinearAxis xAxis;
        private LinearAxis yAxis;

        /// <inheritdoc/>
        public override void Load(IServiceProvider provider)
        {
            view = new PlotView
            {
                Dock = DockStyle.Fill,
            };

            model = new PlotModel();

            xAxis = new LinearAxis
            {
                Position = AxisPosition.Bottom,
                Title = $"Observation Dimension: {dimension1SelectedIndex}",
            };

            yAxis = new LinearAxis
            {
                Position = AxisPosition.Left,
                Title = $"Observation Dimension: {dimension2SelectedIndex}",
            };

            model.Axes.Add(xAxis);
            model.Axes.Add(yAxis);

            view.Model = model;

            statusStrip = new StatusStrip
            {
                Visible = false
            };

            dimension1Label = new ToolStripLabel
            {
                Text = "X Axis Dimension:",
                AutoSize = true
            };

            dimension1ComboBox = new ToolStripComboBox()
            {
                Name = "X Axis Dimension",
                AutoSize = true,
            };

            dimension2Label = new ToolStripLabel
            {
                Text = "Y Axis Dimension:",
                AutoSize = true
            };

            dimension2ComboBox = new ToolStripComboBox()
            {
                Name = "Y Axis Dimension",
                AutoSize = true,
            };

            statusStrip.Items.AddRange(new ToolStripItem[] {
                dimension1Label,
                dimension1ComboBox,
                dimension2Label,
                dimension2ComboBox
            });

            view.MouseClick += new MouseEventHandler(onMouseClick);

            var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));

            if (visualizerService != null)
            {
                visualizerService.AddControl(view);
                visualizerService.AddControl(statusStrip);
            }
        }

        /// <inheritdoc/>
        public override void Show(object value)
        {
            if (value is GaussianObservationsStatistics gaussianObservationsStatistics)
            {
              
                var statesCount = gaussianObservationsStatistics.Means.GetLength(0);
                var observationDimensions = gaussianObservationsStatistics.Means.GetLength(1);

                if (dimension1ComboBox.Items.Count == 0)
                {
                    for (int i = 0; i < observationDimensions; i++)
                    {
                        dimension1ComboBox.Items.Add(i);
                        dimension2ComboBox.Items.Add(i);
                    }

                    dimension1ComboBox.SelectedIndexChanged += dimension1ComboBoxSelectedIndexChanged;
                    dimension1ComboBox.SelectedIndex = dimension1SelectedIndex;

                    dimension2ComboBox.SelectedIndexChanged += dimension2ComboBoxSelectedIndexChanged;
                    dimension2ComboBox.SelectedIndex = dimension2SelectedIndex;
                }

                if (colorList == null)
                {
                    colorList = new List<OxyColor>();
                    for (int i = 0; i < statesCount; i++)
                    {
                        OxyColor color = OxyPalettes.Jet(statesCount).Colors[i];
                        colorList.Add(color);
                    }
                }

                if (allScatterSeries != null)
                {
                    foreach (var scatterSeries in allScatterSeries)
                    {
                        scatterSeries.Points.Clear();
                    }
                }
                else
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

                var batchObservationsCount = gaussianObservationsStatistics.BatchObservations.GetLength(0);
                var offset = BufferData && batchObservationsCount > BufferCount ? batchObservationsCount - BufferCount : 0;
                for (int i = offset; i < batchObservationsCount; i++)
                {
                    var dim1 = gaussianObservationsStatistics.BatchObservations[i, dimension1SelectedIndex];
                    var dim2 = gaussianObservationsStatistics.BatchObservations[i, dimension2SelectedIndex];
                    var state = gaussianObservationsStatistics.InferredMostProbableStates[i];
                    allScatterSeries[(int)state].Points.Add(new ScatterPoint(dim1, dim2, value: state, tag: state));
                }

                for (int i = 0; i < statesCount; i++)
                {
                    var xMean = gaussianObservationsStatistics.Means[i, dimension1SelectedIndex];
                    var yMean = gaussianObservationsStatistics.Means[i, dimension2SelectedIndex];

                    var xVar = gaussianObservationsStatistics.CovarianceMatrices[i, dimension1SelectedIndex, dimension1SelectedIndex];
                    var yVar = gaussianObservationsStatistics.CovarianceMatrices[i, dimension2SelectedIndex, dimension2SelectedIndex];
                    var xyCov = gaussianObservationsStatistics.CovarianceMatrices[i, dimension2SelectedIndex, dimension1SelectedIndex];

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

                    double angle = Math.Atan2(evecs[1, 0], evecs[0, 0]);

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

        private void dimension1ComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (dimension1ComboBox.SelectedIndex != dimension1SelectedIndex)
            {
                dimension1SelectedIndex = dimension1ComboBox.SelectedIndex;
                xAxis.Title = $"Observation Dimension: {dimension1SelectedIndex}";
            }
        }

        private void dimension2ComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (dimension2ComboBox.SelectedIndex != dimension2SelectedIndex)
            {
                dimension2SelectedIndex = dimension2ComboBox.SelectedIndex;
                yAxis.Title = $"Observation Dimension: {dimension2SelectedIndex}";
            }
        }

        private void onMouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                statusStrip.Visible = !statusStrip.Visible;
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
