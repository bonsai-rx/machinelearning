using System;
using System.Reactive;
using System.Linq;
using System.Windows.Forms;
using System.Collections.Generic;

using Bonsai;
using Bonsai.Design;
using Bonsai.ML.Design;

using OxyPlot;
using OxyPlot.Series;

using static TorchSharp.torch;

[assembly: TypeVisualizer(typeof(Bonsai.ML.Torch.LDS.Design.StateVisualizer),
    Target = typeof(Bonsai.ML.Torch.LDS.FilteredState))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.Torch.LDS.Design.StateVisualizer),
    Target = typeof(Bonsai.ML.Torch.LDS.SmoothedState))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.Torch.LDS.Design.StateVisualizer),
    Target = typeof(Bonsai.ML.Torch.LDS.OrthogonalizedState))]
[assembly: TypeVisualizer(typeof(Bonsai.ML.Torch.LDS.Design.StateVisualizer),
    Target = typeof(Bonsai.ML.Torch.LDS.LdsState))]

namespace Bonsai.ML.Torch.LDS.Design;

/// <summary>
/// Provides a visualizer for the state means and covariances from a Kalman filter or smoother.
/// </summary>
public class StateVisualizer : BufferedVisualizer
{
    private TimeSeriesOxyPlotBase _plot;
    private LineSeries[] _lineSeries;
    private AreaSeries[] _areaSeries;

    /// <summary>
    /// Gets or sets the amount of time in seconds that should be shown along the x axis.
    /// </summary>
    public int Capacity { get; set; } = 10;

    /// <summary>
    /// Gets or sets a boolean value that determines whether to buffer the data beyond the capacity.
    /// </summary>
    public bool BufferData { get; set; } = false;

    /// <summary>
    /// Gets the underlying plot control.
    /// </summary>
    public TimeSeriesOxyPlotBase Plot => _plot;

    /// <inheritdoc/>
    public override void Load(IServiceProvider provider)
    {
        _plot = new TimeSeriesOxyPlotBase()
        {
            Capacity = Capacity,
            Dock = DockStyle.Fill,
            StartTime = DateTime.Now,
            BufferData = BufferData
        };

        var capacityStatusLabel = new ToolStripStatusLabel
        {
            Text = "Capacity: ",
            AutoSize = true
        };

        var capacityStatusControl = new ToolStripTextBox
        {
            Text = Capacity.ToString(),
            AutoSize = true
        };

        capacityStatusControl.TextChanged += (sender, e) =>
        {
            if (int.TryParse(capacityStatusControl.Text, out int capacity))
            {
                Capacity = capacity;
                _plot.Capacity = Capacity;
            }
        };

        var bufferDataStatusLabel = new ToolStripStatusLabel
        {
            Text = "Buffer Data: ",
            AutoSize = true
        };

        var bufferDataStatusControl = new ToolStripComboBox
        {
            AutoSize = true
        };

        bufferDataStatusControl.Items.AddRange(["True", "False"]);
        bufferDataStatusControl.SelectedIndex = BufferData ? 0 : 1;

        bufferDataStatusControl.SelectedIndexChanged += (sender, e) =>
        {
            BufferData = bufferDataStatusControl.SelectedIndex == 0;
            _plot.BufferData = BufferData;
        };

        _plot.StatusStrip.Items.Add(capacityStatusLabel);
        _plot.StatusStrip.Items.Add(capacityStatusControl);
        _plot.StatusStrip.Items.Add(bufferDataStatusLabel);
        _plot.StatusStrip.Items.Add(bufferDataStatusControl);

        var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
        visualizerService?.AddControl(_plot);
    }

    /// <inheritdoc/>
    public override void Show(object value)
    {
    }

    /// <inheritdoc/>
    protected override void Show(DateTime time, object value)
    {
        if (value is null) return;

        if (value is not ILdsState state)
            throw new ArgumentException($"Expected value to be a type of {nameof(ILdsState)}.", nameof(value));

        var mean = state.Mean;
        var covariance = state.Covariance;

        if (mean is null || covariance is null) return;

        if (mean.Dimensions == 1)
        {
            mean = mean.unsqueeze(0);
            covariance = covariance.unsqueeze(0);
        }

        var numTimesteps = mean.shape[0];
        var numStates = mean.shape[1];

        if (_lineSeries is null || _areaSeries is null)
        {
            var colors = new OxyColorPresetCycle();

            _lineSeries = new LineSeries[numStates];
            _areaSeries = new AreaSeries[numStates];

            for (int i = 0; i < numStates; i++)
            {
                var currentColor = colors.Next();

                _lineSeries[i] = _plot.AddNewLineSeries(
                    lineSeriesName: $"Mean: {i}",
                    color: currentColor
                );

                _areaSeries[i] = _plot.AddNewAreaSeries(
                    areaSeriesName: $"Variance: {i}",
                    color: currentColor
                );
            }
        }

        var covarianceDiagonal = covariance.diagonal(0, 1, 2);

        for (int i = 0; i < numTimesteps; i++)
        {
            for (int j = 0; j < numStates; j++)
            {

                var meanVal = mean[i, j].to_type(ScalarType.Float64).item<double>();

                _plot.AddToLineSeries(
                    lineSeries: _lineSeries[j],
                    time: time,
                    value: meanVal
                );

                var sigmaVal = covarianceDiagonal[i, j].sqrt().to_type(ScalarType.Float64).item<double>();

                _plot.AddToAreaSeries(
                    areaSeries: _areaSeries[j],
                    time: time,
                    value1: meanVal + sigmaVal,
                    value2: meanVal - sigmaVal
                );
            }
        }
    }

    /// <inheritdoc/>
    protected override void ShowBuffer(IList<Timestamped<object>> values)
    {
        base.ShowBuffer(values);
        if (values.Count > 0)
        {
            var time = values.LastOrDefault().Timestamp.DateTime;
            _plot.SetAxes(minTime: time.AddSeconds(-Capacity), maxTime: time);
            _plot.UpdatePlot();
        }
    }

    /// <inheritdoc/>
    public override void Unload()
    {
        if (!_plot.IsDisposed) _plot.Dispose();
    }
}
