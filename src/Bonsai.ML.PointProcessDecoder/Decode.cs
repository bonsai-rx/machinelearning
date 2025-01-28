using System;
using System.ComponentModel;
using System.Reactive.Linq;
using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Likelihood;
using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Decodes the input neural data into a posterior state estimate using a point process model.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Transform)]
[Description("Decodes the input neural data into a posterior state estimate using a point process model.")]
public class Decode
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Model { get; set; } = string.Empty;

    private bool _ignoreNoSpikes = false;
    private bool _updateIgnoreNoSpikes = false;
    /// <summary>
    /// Gets or sets a value indicating whether to ignore contributions from no spike events.
    /// </summary>
    [Description("Indicates whether to ignore contributions from no spike events.")]
    public bool IgnoreNoSpikes
    {
        get => _ignoreNoSpikes;
        set
        {
            _ignoreNoSpikes = value;
            _updateIgnoreNoSpikes = true;
        }
    }

    /// <summary>
    /// Decodes the input neural data into a posterior state estimate using a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(input => {
            var model = PointProcessModelManager.GetModel(Model);
            if (_updateIgnoreNoSpikes && model.Likelihood is ClusterlessLikelihood likelihood) {
                likelihood.IgnoreNoSpikes = _ignoreNoSpikes;
                _updateIgnoreNoSpikes = false;
            }
            return model.Decode(input);
        });
    }
}