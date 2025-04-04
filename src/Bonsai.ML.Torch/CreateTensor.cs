using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using System.Reactive.Linq;
using System.Reflection;
using System.Xml.Serialization;
using Bonsai.Expressions;
using static TorchSharp.torch;
using Bonsai.ML.Data;
using TorchSharp;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Creates a tensor from the specified values.
    /// Uses Python-like syntax to specify the tensor values. 
    /// For example, a 2x2 tensor can be created with the following values: "[[1, 2], [3, 4]]".
    /// </summary>
    [Combinator]
    [ResetCombinator]
    [Description("Creates a tensor from the specified values. Uses Python-like syntax to specify the tensor values. For example, a 2x2 tensor can be created with the following values: \"[[1, 2], [3, 4]]\".")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class CreateTensor : IScalarTypeProvider
    {
        /// <summary>
        /// The data type of the tensor elements.
        /// </summary>
        [Description("The data type of the tensor elements.")]
        [TypeConverter(typeof(ScalarTypeConverter))]
        public ScalarType Type
        {
            get => _scalarType;
            set
            {
                _scalarType = value;
                _tensor = ConvertTensorScalarType(_tensor, value);
            }
        }
        private ScalarType _scalarType = ScalarType.Float32;

        /// <summary>
        /// The values of the tensor elements. 
        /// Uses Python-like syntax to specify the tensor values.
        /// For example: "[[1, 2], [3, 4]]".
        /// </summary>
        [XmlIgnore]
        [Description("The values of the tensor elements. Uses Python-like syntax to specify the tensor values. For example: \"[[1, 2], [3, 4]]\".")]
        [TypeConverter(typeof(TensorConverter))]
        public Tensor Values
        {
            get => _tensor;
            set => _tensor = ConvertTensorScalarType(value, _scalarType);
        }

        private Tensor _tensor = zeros(1, dtype: ScalarType.Float32);

        /// <summary>
        /// This method converts the tensor to the specified scalar type.
        /// </summary>
        /// <remarks>
        /// We use this method in the setter of the <see cref="Values"/> and <see cref="Type"/> properties to ensure that the tensor is converted to the appropriate type and then returned.
        /// </remarks>
        private static Tensor ConvertTensorScalarType(Tensor value, ScalarType scalarType)
        {
            return value.to_type(scalarType);
        }

        /// <summary>
        /// The values of the tensor elements in XML string format.
        /// </summary>
        [Browsable(false)]
        [XmlElement(nameof(Values))]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public string ValuesXml
        {
            get => TensorConverter.ConvertToString(Values, _scalarType);
            set => Values = TensorConverter.ConvertFromString(value, _scalarType);
        }

        /// <summary>
        /// The device on which to create the tensor.
        /// </summary>
        [XmlIgnore]
        [Description("The device on which to create the tensor.")]
        public Device Device
        {
            get => _device;
            set => _device = value;
        }
        private Device _device = null;

        /// <summary>
        /// Returns an observable sequence that creates a tensor from the specified values.
        /// </summary>
        public IObservable<Tensor> Process()
        {
            return Observable.Return(_device != null ? _tensor.to(_device).clone() : _tensor.clone());
        }

        /// <summary>
        /// Returns an observable sequence that creates a tensor from the specified values for each element in the input sequence.
        /// </summary>
        public IObservable<Tensor> Process<T>(IObservable<T> source)
        {
            var tensor = _tensor.clone();
            return source.Take(1)
                .Select(_ => {
                    tensor = tensor.to(_device);
                    return tensor;
                })
                .Concat(
                    source.Skip(1)
                        .Select(_ => tensor)
                );
        }
    }
}
