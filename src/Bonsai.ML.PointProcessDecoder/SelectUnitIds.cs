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
public class SelectUnitIds : IManagedPointProcessModelNode, IScalarTypeProvider
{
    /// <summary>
    /// The name of the point process model to use.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to use.")]
    public string Name { get; set; } = string.Empty;

    private static readonly ScalarType _scalarType = ScalarType.Int32;
    /// <summary>
    /// The data type of the tensor elements.
    /// </summary>
    [XmlIgnore]
    [Browsable(false)]
    public ScalarType Type => _scalarType;

    private Tensor _unitIds = empty(0, dtype: _scalarType);
    /// <summary>
    /// Gets or sets the selected units.
    /// </summary>
    [XmlIgnore]
    [Description("The selected units IDs.")]
    [TypeConverter(typeof(TensorConverter))]
    public Tensor UnitIds
    {
        get => _unitIds;
        set 
        {
            if (value.Dimensions != 1)
            {
                throw new ArgumentException("Unit IDs must be a 1D tensor.");
            }
            
            _unitIds = value.to_type(_scalarType);
        }
    }

    /// <summary>
    /// The unit IDs in XML string format.
    /// </summary>
    [Browsable(false)]
    [XmlElement(nameof(UnitIds))]
    [EditorBrowsable(EditorBrowsableState.Never)]
    public string UnitIdsXml
    {
        get => TensorConverter.ConvertToString(UnitIds, _scalarType);
        set => UnitIds = TensorConverter.ConvertFromString(value, _scalarType);
    }

    /// <summary>
    /// Decodes the input neural data into a posterior state estimate using a point process model.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Tuple<Tensor, Tensor>> Process(IObservable<Tensor> source)
    {
        var modelName = Name;
        return source.Select(input => 
        {
            var model = PointProcessModelManager.GetModel(modelName);

            var device = input.device;
            var unitIds = _unitIds.clone();

            if (unitIds.device != input.device)
            {
                unitIds = unitIds.to(device);
            }

            return _unitIds.NumberOfElements == 0 ? 
                Tuple.Create(empty(0, device: device), unitIds) :
                Tuple.Create(input[torch.TensorIndex.Colon, torch.TensorIndex.Tensor(unitIds)], unitIds);
        });
    }
}