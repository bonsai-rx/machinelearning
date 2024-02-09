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
    public class TimeSeriesOxyPlotBase : UserControl
    {
        private PlotView view;
        private PlotModel model;
        private ComboBox comboBox;
        private Label label;

        private string lineSeriesName;
        private string areaSeriesName;

        private int selectedIndex;

        private LineSeries lineSeries;
        private AreaSeries areaSeries;
        private Axis xAxis;
        private Axis yAxis;

        public event EventHandler ComboBoxValueChanged;

        private IEnumerable dataSource;

        public DateTime StartTime {get;set;}
        public int Capacity { get; set; }

        public TimeSeriesOxyPlotBase(string _lineSeriesName, string _areaSeriesName, IEnumerable _dataSource, int _selectedIndex)
        {
            lineSeriesName = _lineSeriesName;
            areaSeriesName = _areaSeriesName;
            dataSource = _dataSource;
            selectedIndex = _selectedIndex;
            Initialize();
        }

        public ComboBox ComboBox
        {
            get => comboBox;
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
                Title = lineSeriesName,
                Color = OxyColors.Blue
            };

            areaSeries = new AreaSeries {
                Title = areaSeriesName,
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
                Title = "Value"
            };

            model.Axes.Add(xAxis);
            model.Axes.Add(yAxis);

            model.Series.Add(lineSeries);
            model.Series.Add(areaSeries);

            view.Model = model;

            label = new Label
            {
                Text = "State component:",
                AutoSize = true,
                Location = new Point(-200, 8),
                Anchor = AnchorStyles.Top | AnchorStyles.Right
            };

            comboBox = new ComboBox
            {
                Location = new Point(0, 5),
                DataSource = dataSource,
                Anchor = AnchorStyles.Top | AnchorStyles.Right,
                BindingContext = BindingContext
            };

            comboBox.SelectedIndexChanged += ComboBoxSelectedIndexChanged;
            comboBox.SelectedIndex = selectedIndex;

            Controls.Add(label);
            Controls.Add(comboBox);
            Controls.Add(view);

            comboBox.BringToFront();
            label.BringToFront();

            AutoScaleDimensions = new SizeF(6F, 13F);
        }

        private void ComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            if (comboBox.SelectedIndex != selectedIndex)
            {
                selectedIndex = comboBox.SelectedIndex;
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

        public void Update()
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

            xAxis.Minimum = DateTimeAxis.ToDouble(StartTime);
            xAxis.Maximum = DateTimeAxis.ToDouble(StartTime.AddSeconds(Capacity));
        }
    }
}
