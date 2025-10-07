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

[assembly: TypeVisualizer(typeof(Bonsai.ML.Torch.LDS.Design.ExpectationMaximizationVisualizer),
    Target = typeof(Bonsai.ML.Torch.LDS.ExpectationMaximizationResult))]

namespace Bonsai.ML.Torch.LDS.Design;

/// <summary>
/// Provides a visualizer for the state means and covariances from a Kalman filter or smoother.
/// </summary>
public class ExpectationMaximizationVisualizer : BufferedVisualizer
{
    private TimeSeriesOxyPlotBase _plot;
    private LineSeries _lineSeries;

    /// <summary>
    /// Gets the underlying plot control.
    /// </summary>
    public TimeSeriesOxyPlotBase Plot => _plot;

    /// <inheritdoc/>
    public override void Load(IServiceProvider provider)
    {
        _plot = new TimeSeriesOxyPlotBase()
        {
            Dock = DockStyle.Fill,
            StartTime = DateTime.Now,
            BufferData = true,
            ValueLabel = "Log Likelihood"
        };

        _lineSeries = _plot.AddNewLineSeries("Log Likelihood", OxyColors.Blue);

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

        if (value is not ExpectationMaximizationResult result) return;

        var logLikelihood = result.LogLikelihood;
        if (logLikelihood is null) return;

        var ll = logLikelihood[-1].to_type(ScalarType.Float64).item<double>();

        _plot.AddToLineSeries(
            lineSeries: _lineSeries,
            time: time,
            value: ll
        );
    }

    /// <inheritdoc/>
    protected override void ShowBuffer(IList<Timestamped<object>> values)
    {
        base.ShowBuffer(values);
        if (values.Count > 0)
        {
            _plot.UpdatePlot();
        }
    }

    /// <inheritdoc/>
    public override void Unload()
    {
        if (!_plot.IsDisposed) _plot.Dispose();
    }
}
