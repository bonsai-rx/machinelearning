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
    internal class TimeSeriesOxyPlotBase : UserControl
    {
        private PlotView view;
        private PlotModel model;
        private ToolStripComboBox comboBox;
        private ToolStripLabel label;

        private string _lineSeriesName;
        private string _areaSeriesName;
        private int _selectedIndex;
        private IEnumerable _dataSource;

        private LineSeries lineSeries;
        private AreaSeries areaSeries;
        private Axis xAxis;
        private Axis yAxis;

        private StatusStrip statusStrip;

        /// <summary>
        /// Event handler which can be used to hook into events generated when the combobox values have changed.
        /// </summary>
        public event EventHandler ComboBoxValueChanged;

        /// <summary>
        /// DateTime value that determines the starting time of the data values.
        /// </summary>
        public DateTime StartTime {get;set;}

        /// <summary>
        /// Integer value that determines how many data points should be shown along the x axis.
        /// </summary>
        public int Capacity { get; set; }

        public ToolStripComboBox ComboBox
        {
            get => comboBox;
        }

        public StatusStrip StatusStrip
        {
            get => statusStrip;
        }

        /// <summary>
        /// Constructor of the TimeSeriesOxyPlotBase class.
        /// Requires a line series name and an area series name.
        /// Data source is optional, since pasing it to the constructor will populate the combobox and leave it empty otherwise.
        /// The selected index is only needed when the data source is provided.
        /// </summary>
        public TimeSeriesOxyPlotBase(string lineSeriesName, string areaSeriesName, IEnumerable dataSource = null, int selectedIndex = 0)
        {
            _lineSeriesName = lineSeriesName;
            _areaSeriesName = areaSeriesName;
            _dataSource = dataSource;
            _selectedIndex = selectedIndex;
            Initialize();
        }

        private void Initialize()
        {
            view = new PlotView
            {
                Size = Size,
                Dock = DockStyle.Fill,
            };

            model = new PlotModel();

            lineSeries = new LineSeries {
                Title = _lineSeriesName,
                Color = OxyColors.Blue
            };

            areaSeries = new AreaSeries {
                Title = _areaSeriesName,
                Color = OxyColors.LightBlue,
                Fill = OxyColor.FromArgb(100, 173, 216, 230)
            };

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

            model.Series.Add(lineSeries);
            model.Series.Add(areaSeries);

            view.Model = model;
            Controls.Add(view);

            statusStrip = new StatusStrip
            {
                Visible = false
            };

            if (_dataSource != null)
            {
                InitializeComboBox(_dataSource);

                statusStrip.Items.AddRange(new ToolStripItem[] {
                    label,
                    comboBox,
                });

                view.MouseClick += new MouseEventHandler(onMouseClick);
            }

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

        private void InitializeComboBox(IEnumerable dataSource)
        {
            label = new ToolStripLabel
            {
                Text = "State component:",
                AutoSize = true,
            };

            comboBox = new ToolStripComboBox()
            {
                Name = "stateComponent",
                AutoSize = true,
            };

            foreach (var value in dataSource)
            {
                comboBox.Items.Add(value);
            }
            
            comboBox.SelectedIndexChanged += ComboBoxSelectedIndexChanged;
            comboBox.SelectedIndex = _selectedIndex;
        }

        private void ComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox.SelectedIndex != _selectedIndex)
            {
                _selectedIndex = comboBox.SelectedIndex;
                ComboBoxValueChanged?.Invoke(this, e);
            }
        }

        public void AddToLineSeries(DateTime time, double mean)
        {
            lineSeries.Points.Add(new DataPoint(DateTimeAxis.ToDouble(time), mean));
        }

        public void AddToAreaSeries(DateTime time, double mean, double variance)
        {
            areaSeries.Points.Add(new DataPoint(DateTimeAxis.ToDouble(time), mean + variance));
            areaSeries.Points2.Add(new DataPoint(DateTimeAxis.ToDouble(time), mean - variance));
        }

        public void SetAxes(DateTime minTime, DateTime maxTime)
        {
            xAxis.Minimum = DateTimeAxis.ToDouble(minTime);
            xAxis.Maximum = DateTimeAxis.ToDouble(maxTime);
        }

        public void UpdatePlot()
        {
            model.InvalidatePlot(true);
        }

        public void ResetSeries()
        {
            lineSeries.Points.Clear();
            areaSeries.Points.Clear();
            areaSeries.Points2.Clear();

            xAxis.Reset();
            yAxis.Reset();

            xAxis.Minimum = DateTimeAxis.ToDouble(StartTime.AddSeconds(-Capacity));
            xAxis.Maximum = DateTimeAxis.ToDouble(StartTime);
        }
    }
}
