using System;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using TorchSharp;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch;

/// <summary>
/// Provides a type converter that automatically converts all tensor properties with the <see cref="TensorConverter"/> attribute when the scalar type changes.
/// Apply this converter to classes using [TypeConverter(typeof(TensorOperatorConverter))].
/// </summary>
public class TensorOperatorConverter : TypeConverter
{
    /// <inheritdoc/>
    public override PropertyDescriptorCollection GetProperties(ITypeDescriptorContext context, object value, Attribute[] attributes)
    {
        var properties = TypeDescriptor.GetProperties(value, attributes);
        return new PropertyDescriptorCollection(
            properties.Cast<PropertyDescriptor>()
                .Select(p => p.Name == nameof(IScalarTypeProvider.Type) 
                    ? new ScalarTypePropertyDescriptor(p) 
                    : p)
                .ToArray());
    }

    /// <inheritdoc/>
    public override bool GetPropertiesSupported(ITypeDescriptorContext context) => true;

    private class ScalarTypePropertyDescriptor(PropertyDescriptor baseDescriptor) : PropertyDescriptor(baseDescriptor)
    {
        private readonly PropertyDescriptor _baseDescriptor = baseDescriptor;
        public override Type ComponentType => _baseDescriptor.ComponentType;
        public override bool IsReadOnly => _baseDescriptor.IsReadOnly;
        public override Type PropertyType => _baseDescriptor.PropertyType;

        public override bool CanResetValue(object component) => _baseDescriptor.CanResetValue(component);
        public override object GetValue(object component) => _baseDescriptor.GetValue(component);
        public override void ResetValue(object component) => _baseDescriptor.ResetValue(component);
        public override bool ShouldSerializeValue(object component) => _baseDescriptor.ShouldSerializeValue(component);

        public override void SetValue(object component, object value)
        {
            var oldValue = _baseDescriptor.GetValue(component);
            if (Equals(oldValue, value))
                return;

            _baseDescriptor.SetValue(component, value);

            if (value is ScalarType newScalarType && component is IScalarTypeProvider)
            {
                ConvertAllTensorProperties(component, newScalarType);
            }
        }

        private static void ConvertAllTensorProperties(object component, ScalarType scalarType)
        {
            var properties = TypeDescriptor.GetProperties(component);
            foreach (PropertyDescriptor property in properties)
            {
                // Check if this property uses TensorConverter
                var converterAttr = property.Attributes.OfType<TypeConverterAttribute>().FirstOrDefault();
                
                if (converterAttr?.ConverterTypeName?.Contains(nameof(TensorConverter)) != true)
                    continue;

                if (property.PropertyType != typeof(Tensor))
                    continue;

                if (property.GetValue(component) is not Tensor tensor || tensor.dtype == scalarType)
                    continue;

                property.SetValue(component, tensor.to_type(scalarType));
            }
        }
    }
}
