using System;
using System.Windows.Forms;
using System.Collections.Generic;
using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Design;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using OxyPlot.WindowsForms;

[assembly: TypeVisualizer(typeof(Bonsai.ML.HiddenMarkovModels.Design.GaussianObservationsStatisticsClustersVisualizer),
    Target = typeof(Bonsai.ML.HiddenMarkovModels.Observations.GaussianObservationsStatistics))]

namespace Bonsai.ML.HiddenMarkovModels.Design
{
    /// <summary>
    /// Provides a type visualizer of <see cref="Observations.GaussianObservationsStatistics"/> to display how the observations 
    /// cluster with respect to the mean and covariance of each state of an HMM with gaussian observations model.
    /// </summary>
    public class GaussianObservationsStatisticsClustersVisualizer : DialogTypeVisualizer
    {
        private PlotView view;
        private PlotModel model;
        private List<LineSeries> allLineSeries = null;
        private List<ScatterSeries> allScatterSeries = null;
        private List<OxyColor> colorList = null;
        private StatusStrip statusStrip;
        private int dimension1SelectedIndex = 0;
        private ToolStripComboBox dimension1ComboBox;
        private ToolStripLabel dimension1Label;
        private int dimension2SelectedIndex = 1;
        private ToolStripComboBox dimension2ComboBox;
        private ToolStripLabel dimension2Label;
        private LinearAxis xAxis;
        private LinearAxis yAxis;

        /// <summary>
        /// Gets the status strip.
        /// </summary>
        public StatusStrip StatusStrip => statusStrip;

        /// <summary>
        /// Gets the selected index of the first dimension.
        /// </summary>
        public int Dimension1SelectedIndex { get => dimension1SelectedIndex; set => dimension1SelectedIndex = value; }

        /// <summary>
        /// Gets the selected index of the second dimension.
        /// </summary>
        public int Dimension2SelectedIndex { get => dimension2SelectedIndex; set => dimension2SelectedIndex = value; }

        /// <summary>
        /// Gets the first dimension combo box.
        /// </summary>
        public ToolStripComboBox Dimension1ComboBox => dimension1ComboBox;

        /// <summary>
        /// Gets the second dimension combo box.
        /// </summary>
        public ToolStripComboBox Dimension2ComboBox => dimension2ComboBox;

        /// <summary>
        /// Gets or sets a value indicating whether the data should be buffered.
        /// </summary>
        public bool BufferData { get; set; } = true;

        /// <summary>
        /// Gets or sets the buffer count.
        /// </summary>
        public int BufferCount { get; set; } = 250;

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
            if (value is Observations.GaussianObservationsStatistics gaussianObservationsStatistics)
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
                    dimension2SelectedIndex = Math.Max(dimension2ComboBox.Items.Count - 1, dimension2SelectedIndex);
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
                var predictedStatesCount = gaussianObservationsStatistics.PredictedStates.Length;

                for (int i = offset; i < batchObservationsCount; i++)
                {
                    var dim1 = gaussianObservationsStatistics.BatchObservations[i, dimension1SelectedIndex];
                    var dim2 = gaussianObservationsStatistics.BatchObservations[i, dimension2SelectedIndex];
                    var state = gaussianObservationsStatistics.PredictedStates[i];
                    allScatterSeries[Convert.ToInt32(state)].Points.Add(new ScatterPoint(dim1, dim2, value: state, tag: state));
                }

                for (int i = 0; i < statesCount; i++)
                {
                    var xMean = gaussianObservationsStatistics.Means[i, dimension1SelectedIndex];
                    var yMean = gaussianObservationsStatistics.Means[i, dimension2SelectedIndex];

                    var xVar = gaussianObservationsStatistics.CovarianceMatrices[i, dimension1SelectedIndex, dimension1SelectedIndex];
                    var yVar = gaussianObservationsStatistics.CovarianceMatrices[i, dimension2SelectedIndex, dimension2SelectedIndex];
                    var xyCov = gaussianObservationsStatistics.CovarianceMatrices[i, dimension2SelectedIndex, dimension1SelectedIndex];

                    var ellipseParameters = EllipseHelper.GetEllipseParameters(xVar, yVar, xyCov);

                    for (int j = 1; j < 4; j++)
                    {
                        var points = new List<DataPoint>();
                        int numPoints = 100;
                        for (int k = 0; k < numPoints + 1; k++)
                        {
                            double theta = 2 * Math.PI * k / numPoints;
                            double x = j * ellipseParameters.MajorAxis * Math.Cos(theta);
                            double y = j * ellipseParameters.MinorAxis * Math.Sin(theta);

                            double xRot = x * Math.Cos(ellipseParameters.Angle) - y * Math.Sin(ellipseParameters.Angle);
                            double yRot = x * Math.Sin(ellipseParameters.Angle) + y * Math.Cos(ellipseParameters.Angle);

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
