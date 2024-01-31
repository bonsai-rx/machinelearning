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

        public DateTime? StartTime {get;set;}
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

            xAxis = new LinearAxis {
                Position = AxisPosition.Bottom,
                Title = "Time",
                StringFormat = "0",
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                MinimumMajorStep = 1,
                MinimumMinorStep = 0.5,
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
                AutoSize = true,
                DataSource = dataSource,
                Anchor = AnchorStyles.Top | AnchorStyles.Right
            };

            comboBox.SelectedIndexChanged += ComboBoxDataSourceInitialized;

            Controls.Add(label);
            Controls.Add(comboBox);
            Controls.Add(view);

            comboBox.BringToFront();
            label.BringToFront();

            AutoScaleDimensions = new SizeF(6F, 13F);
        }

        private void ComboBoxDataSourceInitialized(object sender, EventArgs e)
        {
            comboBox.SelectedIndex = selectedIndex;
            comboBox.SelectedIndexChanged -= ComboBoxDataSourceInitialized;
            comboBox.SelectedIndexChanged += ComboBoxSelectedIndexChanged;
        }

        private void ComboBoxSelectedIndexChanged(object sender, EventArgs e)
        {
            ComboBoxValueChanged?.Invoke(this, e);
        }

        public void AddToLineSeries(double time, double mean)
        {
            lineSeries.Points.Add(new DataPoint(time, mean));
        }

        public void AddToAreaSeries(double time, double mean, double variance)
        {
            areaSeries.Points.Add(new DataPoint(time, mean + variance));
            areaSeries.Points2.Add(new DataPoint(time, mean - variance));
        }

        public void SetAxes(double minTime, double maxTime)
        {
            xAxis.Minimum = minTime;
            xAxis.Maximum = maxTime;
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

            xAxis.Minimum = 0;
            xAxis.Maximum = Capacity;
        }
    }
}
