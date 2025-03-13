using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.ComponentModel;
using static TorchSharp.torch;
using OpenCV.Net;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Represents a type converter for data types that are currently supported between .NET types and torch <see cref="ScalarType"/>.
    /// It is designed to be used in the <see cref="TypeConverterAttribute"/> of properties that represent <see cref="ScalarType"/> or <see cref="Type"/>.
    /// </summary>
    public class ScalarTypeConverter : TypeConverter
    {
        /// <inheritdoc/>
        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
        {
            if (sourceType == typeof(string))
                return true;
            if (sourceType == typeof(Type))
                return true;
            if (sourceType == typeof(ScalarType))
                return true;

            return base.CanConvertFrom(context, sourceType);
        }

        /// <inheritdoc/>
        public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
        {
            if (destinationType == typeof(string))
                return true;
            if (destinationType == typeof(Type))
                return true;
            if (destinationType == typeof(ScalarType))
                return true;

            return base.CanConvertTo(context, destinationType);
        }

        /// <inheritdoc/>
        // We currently only support a subset of all possible ScalarTypes when converting to/from .NET types using the ScalarTypeLookup class.
        // Properties of type `ScalarType` that use this converter will only display the ScalarTypes that are currently supported for conversion.
        // Those properties will display the ScalarType enum value represented in string format.
        // Properties of type `Type` that use this converter will display the .NET types that are currently supported.
        // These properties will use the corresponding string value in the lookup table for display.
        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            // Get the property type we are converting from
            var propType = context?.PropertyDescriptor?.PropertyType;

            // Check if we are converting from a string or Type to ScalarType
            if (propType == typeof(ScalarType))
            {
                // Convert from string to ScalarType
                if (value is string stringValue)
                {
                    foreach (var scalarType in ScalarTypeLookup.ScalarTypes)
                    {
                        if (string.Equals(scalarType.ToString(), stringValue, StringComparison.OrdinalIgnoreCase))
                            return scalarType;
                    }
                    throw new ArgumentException($"'{value}' is not a supported ScalarType.");
                }

                // Convert from Type to ScalarType
                else if (value is Type type)
                {
                    if (ScalarTypeLookup.TryGetScalarTypeFromType(type, out var found))
                        return found;
                    throw new ArgumentException($"'{value}' is not a supported ScalarType.");
                }
            }

            // Check if we are converting from a string or ScalarType to Type            
            else if (propType == typeof(Type))
            {
                // Convert from string to Type
                if (value is string stringValue)
                {
                    if (ScalarTypeLookup.TryGetScalarTypeFromString(stringValue, out var found))
                        return ScalarTypeLookup.GetTypeFromScalarType(found);
                    throw new ArgumentException($"'{value}' is not a supported Type.");
                }

                // Convert from ScalarType to Type
                else if (value is ScalarType scalarType)
                {
                    if (ScalarTypeLookup.TryGetTypeFromScalarType(scalarType, out var found))
                        return found;
                    throw new ArgumentException($"'{value}' is not a supported Type.");
                }
            }

            return base.ConvertFrom(context, culture, value);
        }

        /// <inheritdoc/>
        // When converting to a designated type, we will convert the value based on whether the property type is `ScalarType` or `Type`.
        // If the property type is `ScalarType`, we will convert the value to a string representation of the ScalarType enum.
        // If the property type is `Type`, we will convert the value to the corresponding string representation in the lookup table.
        public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            var propType = context?.PropertyDescriptor?.PropertyType;

            if (propType == typeof(ScalarType) && value is ScalarType scalarType)
            {
                if (destinationType == typeof(string))
                {
                    foreach (var scalarTypeLookup in ScalarTypeLookup.ScalarTypes)
                    {
                        if (string.Equals(scalarType.ToString(), scalarTypeLookup.ToString(), StringComparison.OrdinalIgnoreCase))
                            return scalarType.ToString();
                    }
                }

                else if (destinationType == typeof(Type))
                {
                    if (ScalarTypeLookup.TryGetTypeFromScalarType(scalarType, out var found))
                        return found;
                }
            }

            else if (propType == typeof(Type) && value is Type type)
            {
                if (destinationType == typeof(string))
                {
                    if (ScalarTypeLookup.TryGetStringFromType(type, out var found))
                        return found;
                }
                
                else if (destinationType == typeof(ScalarType))
                {
                    if (ScalarTypeLookup.TryGetScalarTypeFromType(type, out var found))
                        return found;
                }
            }

            return base.ConvertTo(context, culture, value, destinationType);
        }

        /// <inheritdoc/>
        public override bool GetStandardValuesSupported(ITypeDescriptorContext context) => true;

        /// <inheritdoc/>
        public override bool GetStandardValuesExclusive(ITypeDescriptorContext context) => true;

        /// <inheritdoc/>
        // We will return a collection of supported ScalarTypes if the property type is `ScalarType` or a collection of supported .NET types if the property type is `Type`.
        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            if (context?.PropertyDescriptor?.PropertyType == typeof(ScalarType))
                return new StandardValuesCollection(ScalarTypeLookup.ScalarTypes.ToArray());

            else if (context?.PropertyDescriptor?.PropertyType == typeof(Type))
                return new StandardValuesCollection(ScalarTypeLookup.Types.ToArray());

            return base.GetStandardValues(context);
        }
    }
}