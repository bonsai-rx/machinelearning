using System;
using System.Linq;
using System.Globalization;
using System.ComponentModel;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Represents a type converter for data types that are currently supported between .NET types and torch <see cref="ScalarType"/>.
    /// </summary>
    public class ScalarTypeConverter : TypeConverter
    {
        /// <inheritdoc/>
        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
        {
            if (sourceType == typeof(string) || sourceType == typeof(ScalarType))
                return true;

            return base.CanConvertFrom(context, sourceType);
        }

        /// <inheritdoc/>
        public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
        {
            if (destinationType == typeof(string) || destinationType == typeof(ScalarType))
                return true;

            return base.CanConvertTo(context, destinationType);
        }

        /// <inheritdoc/>
        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            if (value is string stringValue)
            {
                foreach (var scalarType in ScalarTypeLookup.ScalarTypes)
                {
                    if (string.Equals(scalarType.ToString(), stringValue, StringComparison.OrdinalIgnoreCase))
                        return scalarType;
                }
                throw new ArgumentException($"'{value}' is not a supported ScalarType.");
            }

            return base.ConvertFrom(context, culture, value);
        }

        /// <inheritdoc/>
        public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            if (value is ScalarType scalarType)
            {
                if (destinationType == typeof(string))
                {
                    foreach (var scalarTypeLookup in ScalarTypeLookup.ScalarTypes)
                    {
                        if (string.Equals(scalarType.ToString(), scalarTypeLookup.ToString(), StringComparison.OrdinalIgnoreCase))
                            return scalarType.ToString();
                    }
                }
            }

            return base.ConvertTo(context, culture, value, destinationType);
        }

        /// <inheritdoc/>
        public override bool GetStandardValuesSupported(ITypeDescriptorContext context) => true;

        /// <inheritdoc/>
        public override bool GetStandardValuesExclusive(ITypeDescriptorContext context) => true;

        /// <inheritdoc/>
        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            if (context?.PropertyDescriptor?.PropertyType == typeof(ScalarType))
                return new StandardValuesCollection(ScalarTypeLookup.ScalarTypes.ToArray());

            return base.GetStandardValues(context);
        }
    }
}