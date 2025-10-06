using System;
using System.Linq;
using System.Globalization;
using System.ComponentModel;
using static TorchSharp.torch;

using Bonsai.ML.Data;

namespace Bonsai.ML.Torch;

/// <summary>
/// Represents a type converter for tensors to and from string.
/// </summary>
public class TensorConverter : TypeConverter
{
    /// <inheritdoc/>
    public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
    {
        if (sourceType == typeof(string) || sourceType == typeof(Tensor)) return true;
        return base.CanConvertFrom(context, sourceType);
    }

    /// <inheritdoc/>
    public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
    {
        if (destinationType == typeof(string) || destinationType == typeof(Tensor)) return true;
        return base.CanConvertTo(context, destinationType);
    }

    /// <inheritdoc/>
    public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
    {
        if (context?.Instance is IScalarTypeProvider scalarTypeProvider && value is string stringValue)
        {
            var scalarType = scalarTypeProvider.Type;
            return ConvertFromString(stringValue, scalarType);
        }

        return base.ConvertFrom(context, culture, value);
    }

    /// <summary>
    /// Method that converts a string into a tensor. The string value should be in a Python-like syntax. For example, a 2x2 tensor can be created with the following values: "[[1, 2], [3, 4]]".
    /// </summary>
    /// <param name="value">The string value to convert.</param>
    /// <param name="scalarType">The scalar type of the tensor.</param>
    /// <returns>A tensor created from the string value.</returns>
    /// <exception cref="ArgumentException">Thrown when the tensor data is not supported for converting to .NET data type</exception>
    /// <remarks>
    /// This class relies on the PythonDataHelper class to parse the string into the appropriate data type.
    /// The Parse method however returns a type of object, so we use pattern matching to determine the type of the data and create the tensor from the typed object.
    /// </remarks>
    public static Tensor ConvertFromString(string value, ScalarType scalarType)
    {
        var returnType = ScalarTypeLookup.GetTypeFromScalarType(scalarType);

        if (string.IsNullOrEmpty(value))
        {
            return null;
        }

        var tensorData = PythonDataHelper.Parse(value, returnType);

        if (tensorData is Array arrayData)
        {
            return from_array(arrayData, scalarType);
        }
        else if (tensorData is byte byteData)
        {
            return tensor(byteData, scalarType);
        }
        else if (tensorData is short shortData)
        {
            return tensor(shortData, scalarType);
        }
        else if (tensorData is int intData)
        {
            return tensor(intData, scalarType);
        }
        else if (tensorData is long longData)
        {
            return tensor(longData, scalarType);
        }
        else if (tensorData is float floatData)
        {
            return tensor(floatData, scalarType);
        }
        else if (tensorData is double doubleData)
        {
            return tensor(doubleData, scalarType);
        }
        else if (tensorData is bool boolData)
        {
            return tensor(boolData, scalarType);
        }
        else if (tensorData is sbyte sbyteData)
        {
            return tensor(sbyteData, scalarType);
        }
        else
        {
            throw new ArgumentException($"'{tensorData}' is not a supported tensor data type.");
        }
    }

    /// <inheritdoc/>
    public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
    {
        if (value is Tensor tensor && 
            context?.Instance is IScalarTypeProvider scalarTypeProvider &&
            destinationType == typeof(string))
        {
            var scalarType = scalarTypeProvider.Type;
            return ConvertToString(tensor, scalarType);
        }

        return base.ConvertTo(context, culture, value, destinationType);
    }

    /// <summary>
    /// Method that converts a tensor into a string. The string value is in a Python-like syntax. For example, a 2x2 tensor can be created with the following values: "[[1, 2], [3, 4]]".
    /// </summary>
    /// <param name="tensor">The tensor to convert.</param>
    /// <param name="scalarType">The scalar type of the tensor.</param>
    /// <returns>A string representation of the tensor.</returns>
    /// <exception cref="ArgumentException">Thrown when the <see cref="ScalarType"/> is not supported for converting to .NET data type</exception>
    /// <remarks>
    /// The TorchSharp library does not provide a generic method to extract the data from the tensor as an object.
    /// Thus, we use the provided <see cref="ScalarType"/> as the expected .NET data type to extract the data from the tensor.
    /// The data is then passed to the PythonDataHelper class to format the data into the appropriate string representation for the given data type.
    /// </remarks>
    public static string ConvertToString(Tensor tensor, ScalarType scalarType)
    {
        object tensorData;

        if (tensor is null)
            return string.Empty;

        if (tensor.Dimensions == 0)
        {
            if (scalarType == ScalarType.Byte)
            {
                tensorData = tensor.item<byte>();
            }
            else if (scalarType == ScalarType.Int16)
            {
                tensorData = tensor.item<short>();
            }
            else if (scalarType == ScalarType.Int32)
            {
                tensorData = tensor.item<int>();
            }
            else if (scalarType == ScalarType.Int64)
            {
                tensorData = tensor.item<long>();
            }
            else if (scalarType == ScalarType.Float32)
            {
                tensorData = tensor.item<float>();
            }
            else if (scalarType == ScalarType.Float64)
            {
                tensorData = tensor.item<double>();
            }
            else if (scalarType == ScalarType.Bool)
            {
                tensorData = tensor.item<bool>();
            }
            else if (scalarType == ScalarType.Int8)
            {
                tensorData = tensor.item<sbyte>();
            }
            else
            {
                throw new ArgumentException($"'{scalarType}' is not a supported tensor data type.");
            }
        }
        else
        {
            if (scalarType == ScalarType.Byte)
            {
                tensorData = tensor.data<byte>().ToNDArray();
            }
            else if (scalarType == ScalarType.Int16)
            {
                tensorData = tensor.data<short>().ToNDArray();
            }
            else if (scalarType == ScalarType.Int32)
            {
                tensorData = tensor.data<int>().ToNDArray();
            }
            else if (scalarType == ScalarType.Int64)
            {
                tensorData = tensor.data<long>().ToNDArray();
            }
            else if (scalarType == ScalarType.Float32)
            {
                tensorData = tensor.data<float>().ToNDArray();
            }
            else if (scalarType == ScalarType.Float64)
            {
                tensorData = tensor.data<double>().ToNDArray();
            }
            else if (scalarType == ScalarType.Bool)
            {
                tensorData = tensor.data<bool>().ToNDArray();
            }
            else if (scalarType == ScalarType.Int8)
            {
                tensorData = tensor.data<sbyte>().ToNDArray();
            }
            else
            {
                throw new ArgumentException($"'{scalarType}' is not a supported tensor data type.");
            }
        }

        return PythonDataHelper.Format(tensorData);
    }
}
