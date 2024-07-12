using System.Windows.Forms;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.WindowsForms;
using System.Drawing;
using System;
using OxyPlot.Axes;
using System.Collections;

namespace Bonsai.ML.Visualizers
{
    /// <summary>
    /// Provides a user control to display time series data using OxyPlot.
    /// </summary>
    public class TimeSeriesOxyPlotBase : UserControl
    {
        private readonly PlotView view;
        private readonly PlotModel model;
        private static readonly OxyColor defaultLineSeriesColor = OxyColors.Blue;
        private static readonly OxyColor defaultAreaSeriesColor = OxyColors.LightBlue;

        private readonly Axis xAxis;
        private readonly Axis yAxis;

        private readonly StatusStrip statusStrip;

        /// <summary>
        /// Gets or sets the datetime value that determines the starting time of the data values.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the integer value that determines how many data points should be shown along the x axis.
        /// </summary>
        public int Capacity { get; set; }

        /// <summary>
        /// Gets the status strip control.
        /// </summary>
        public StatusStrip StatusStrip => statusStrip;

        /// <summary>
        /// Gets or sets a boolean value that determines whether to buffer the data beyond the capacity.
        /// </summary>
        public bool BufferData { get; set; }

        /// <summary>
        /// Gets or sets the label of the value axis in the time series plot.
        /// </summary>
        public string ValueLabel
        {
            get => yAxis.Title;
            set => yAxis.Title = value;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="TimeSeriesOxyPlotBase"/> class
        /// </summary>
        public TimeSeriesOxyPlotBase()
        {
            view = new PlotView
            {
                Size = Size,
                Dock = DockStyle.Fill,
            };

            model = new PlotModel();

            xAxis = new DateTimeAxis {
                Position = AxisPosition.Bottom,
                Title = "Time",
                StringFormat = "HH:mm:ss",
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                MinorIntervalType = DateTimeIntervalType.Auto,
                IntervalType = DateTimeIntervalType.Auto,
                FontSize = 9
            };

            yAxis = new LinearAxis {
                Position = AxisPosition.Left,
                Title = "Value",
            };

            model.Axes.Add(xAxis);
            model.Axes.Add(yAxis);

            view.Model = model;
            Controls.Add(view);

            statusStrip = new StatusStrip
            {
                Visible = false
            };

            view.MouseClick += new MouseEventHandler(onMouseClick);
            Controls.Add(statusStrip);

            Controls.Add(statusStrip);

            AutoScaleDimensions = new SizeF(6F, 13F);
        }

        private void onMouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                statusStrip.Visible = !statusStrip.Visible;
            }
        }

        /// <summary>
        /// Method to add a combobox with a label to the status strip.
        /// Requires a string label, an enumerable data source, a selected index, and a callback method for the selected index changed event.
        /// </summary>
        public void AddComboBoxWithLabel(string label, IEnumerable dataSource, int selectedIndex, EventHandler onComboBoxSelectionChanged)
        {
            ToolStripLabel toolStripLabel = new ToolStripLabel(label);
            ToolStripComboBox toolStripComboBox = new ToolStripComboBox();

            foreach (var value in dataSource)
            {
                toolStripComboBox.Items.Add(value);
            }

            toolStripComboBox.SelectedIndexChanged += onComboBoxSelectionChanged;
            toolStripComboBox.SelectedIndex = selectedIndex;

            statusStrip.Items.AddRange(new ToolStripItem[] {
                toolStripLabel,
                toolStripComboBox
            });
       
        }

        /// <summary>
        /// Method to add a new line series to the data plot.
        /// Requires a string for the name of the line series
        /// Color of the line series is optional.
        /// </summary>
        public LineSeries AddNewLineSeries(string lineSeriesName, OxyColor? color = null)
        {
            OxyColor _color = color.HasValue ? color.Value : defaultLineSeriesColor;
            LineSeries lineSeries = new LineSeries {
                Title = lineSeriesName,
                Color = _color
            };
            model.Series.Add(lineSeries);
            return lineSeries;
        }

        /// <summary>
        /// Method to add a new area series to the data plot.
        /// Requires a string for the name of the area series
        /// Optional parameters are color of the lines, fill color, and opacity.
        /// Returns the new line series.
        /// </summary>
        public AreaSeries AddNewAreaSeries(string areaSeriesName, OxyColor? color = null, OxyColor? fill = null, byte opacity = 100)
        {
            OxyColor _color = color.HasValue ? color.Value : defaultAreaSeriesColor;
            OxyColor _fill = fill.HasValue? fill.Value : OxyColor.FromArgb(opacity, _color.R, _color.G, _color.B);
            AreaSeries areaSeries = new AreaSeries {
                Title = areaSeriesName,
                Color = _color,
                Fill = _fill
            };
            model.Series.Add(areaSeries);
            return areaSeries;
        }

        /// <summary>
        /// Method to add data to a line series.
        /// Requires the line series, time, and value.
        /// </summary>
        public void AddToLineSeries(LineSeries lineSeries, DateTime time, double value)
        {
            if (!BufferData)
            {
                var timeCap = DateTimeAxis.ToDouble(time.AddSeconds(-Capacity));
                lineSeries.Points.RemoveAll(dataPoint => dataPoint.X < timeCap);
            }
            lineSeries.Points.Add(new DataPoint(DateTimeAxis.ToDouble(time), value));
        }

        /// <summary>
        /// Method to add data to an area series.
        /// Requires the area series, time, value1, and value2.
        /// </summary>
        public void AddToAreaSeries(AreaSeries areaSeries, DateTime time, double value1, double value2)
        {
            if (!BufferData)
            {
                var timeCap = DateTimeAxis.ToDouble(time.AddSeconds(-Capacity));
                areaSeries.Points.RemoveAll(dataPoint => dataPoint.X < timeCap);
                areaSeries.Points2.RemoveAll(dataPoint => dataPoint.X < timeCap);
            }
            var curTime = DateTimeAxis.ToDouble(time);
            areaSeries.Points.Add(new DataPoint(curTime, value1));
            areaSeries.Points2.Add(new DataPoint(curTime, value2));
        }

        /// <summary>
        /// Set the minimum and maximum values to show along the x axis.
        /// Requires the minTime and maxTime.
        /// </summary>
        public void SetAxes(DateTime minTime, DateTime maxTime)
        {
            xAxis.Minimum = DateTimeAxis.ToDouble(minTime);
            xAxis.Maximum = DateTimeAxis.ToDouble(maxTime);
        }

        /// <summary>
        /// Method to update the plot.
        /// </summary>
        public void UpdatePlot()
        {
            model.InvalidatePlot(true);
        }

        /// <summary>
        /// Method to reset the line series.
        /// </summary>
        public void ResetLineSeries(LineSeries lineSeries)
        {
            lineSeries.Points.Clear();
        }

        /// <summary>
        /// Method to reset the area series.
        /// </summary>
        public void ResetAreaSeries(AreaSeries areaSeries)
        {
            areaSeries.Points.Clear();
            areaSeries.Points2.Clear();
        }

        /// <summary>
        /// Method to reset all series in the current PlotModel.
        /// </summary>
        public void ResetModelSeries()
        {
            model.Series.Clear();
        }

        /// <summary>
        /// Method to reset the x and y axes to their default.
        /// </summary>
        public void ResetAxes()
        {
            xAxis.Reset();
            yAxis.Reset();

            xAxis.Minimum = DateTimeAxis.ToDouble(StartTime.AddSeconds(-Capacity));
            xAxis.Maximum = DateTimeAxis.ToDouble(StartTime);
        }
    }
}
