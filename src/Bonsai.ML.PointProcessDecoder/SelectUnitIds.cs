using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Xml.Serialization;
using static TorchSharp.torch;
using Bonsai.ML.Torch;
using OpenCV.Net;
using TorchSharp;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Selects specific units IDs from the input neural data.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Transform)]
[Description("Selects specific units IDs from the input neural data.")]
public class SelectUnitIds : IManagedPointProcessModelNode
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Name { get; set; } = string.Empty;

    private Tensor _unitIds = empty(0);
    /// <summary>
    /// Gets or sets the selected units.
    /// </summary>
    [TypeConverter(typeof(TensorConverter))]
    public Tensor UnitIds
    {
        get => _unitIds;
        set => _unitIds = value.to_type(ScalarType.Int32);
    }

    /// <summary>
    /// The unit IDs in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(UnitIds))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string UnitIdsXml
    {
        get => TensorConverter.ConvertToString(UnitIds, ScalarType.Int32);
        set => UnitIds = TensorConverter.ConvertFromString(value, ScalarType.Int32);
    }

    /// <summary>
    /// Decodes the input neural data into a posterior state estimate using a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, Tensor>> Process(IObservable<Tensor> source)
    {
        return source.Select(input => 
        {
            var device = input.device;

            if (_unitIds.device != input.device)
            {
                _unitIds = _unitIds.to(device);
            }

            return _unitIds.NumberOfElements == 0 ? 
                Tuple.Create(empty(0, device: device), _unitIds) :
                Tuple.Create(input[torch.TensorIndex.Colon, torch.TensorIndex.Tensor(_unitIds)], _unitIds);
        });
    }
}