using Bonsai.ML.Design;

namespace Bonsai.ML.PointProcessDecoder.Design;

/// <summary>
/// Interface for visualizing the output of a point process decoder.
/// </summary>
public interface IDecoderVisualizer
{
    /// <summary>
    /// Gets or sets the capacity of the visualizer.
    /// </summary>
    public int Capacity { get; set; }

    /// <summary>
    /// Gets the heatmap that visualizes a component's output.
    /// </summary>
    public HeatMapSeriesOxyPlotBase Plot { get; }
}
