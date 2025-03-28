using System;
using System.ComponentModel;
using System.Reactive.Linq;
using System.Collections.Generic;
using System.Xml.Serialization;
using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Decodes the input neural data into a posterior state estimate using a point process model.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Transform)]
[Description("Decodes the input neural data into a posterior state estimate using a point process model.")]
public class SelectUnits : IManagedPointProcessModelNode
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Name { get; set; } = string.Empty;

    private Tensor _unitIds = empty(0);
    private List<int> _units = [];
    /// <summary>
    /// Gets or sets the selected units.
    /// </summary>
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public int[] Units
    {
        get => [.._units];
        set
        {
            _units = [..value];
            _unitIds = tensor(value);
        }
    }

    /// <summary>
    /// Gets the selected units in the form of a list.
    /// </summary>
    [Browsable(false)]
    [XmlIgnore]
    public List<int> SelectedUnits => _units;

    /// <summary>
    /// Decodes the input neural data into a posterior state estimate using a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(input => 
        {
            if (_unitIds.NumberOfElements == 0)
            {
                return input;
            }

            if (_unitIds.device != input.device)
            {
                _unitIds = _unitIds.to(input.device);
            }

            return input[TensorIndex.Colon, TensorIndex.Tensor(_unitIds)];
        });
    }
}