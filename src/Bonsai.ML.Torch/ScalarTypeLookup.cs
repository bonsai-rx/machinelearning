using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Provides methods to look up tensor data types that currently support conversion to designated .NET types.
    /// </summary>
    internal static class ScalarTypeLookup
    {
        private static readonly Dictionary<ScalarType, (Type Type, string StringValue)> _lookup = new()
        {
            { ScalarType.Byte, (typeof(byte), "Byte") },
            { ScalarType.Int16, (typeof(short), "Int16") },
            { ScalarType.Int32, (typeof(int), "Int32") },
            { ScalarType.Int64, (typeof(long), "Int64") },
            { ScalarType.Float32, (typeof(float), "Single") },
            { ScalarType.Float64, (typeof(double), "Double") },
            { ScalarType.Bool, (typeof(bool), "Boolean") },
            { ScalarType.Int8, (typeof(sbyte), "SByte") },
        };

        /// <summary>
        /// Returns the currently supported tensor data types.
        /// </summary>
        public static IEnumerable<ScalarType> ScalarTypes => _lookup.Keys;

        /// <summary>
        /// Returns the currently supported .NET data types.
        /// </summary>
        public static IEnumerable<Type> Types => _lookup.Values.Select(x => x.Type);

        /// <summary>
        /// Tries to get the type corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="scalarType"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool TryGetTypeFromScalarType(ScalarType scalarType, out Type type) => _lookup.TryGetValue(scalarType, out (Type Type, string StringValue) value) ? (type = value.Type) != null : (type = null) != null;

        /// <summary>
        /// Returns the type corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static Type GetTypeFromScalarType(ScalarType type) => _lookup[type].Type;

        /// <summary>
        /// Tries to get the string representation corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="scalarType"></param>
        /// <param name="stringValue"></param>
        /// <returns></returns>
        public static bool TryGetStringFromScalarType(ScalarType scalarType, out string stringValue) => _lookup.TryGetValue(scalarType, out (Type Type, string StringValue) value) ? (stringValue = value.StringValue) != null : (stringValue = null) != null;

        /// <summary>
        /// Returns the string representation corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string GetStringFromScalarType(ScalarType type) => _lookup[type].StringValue;

        /// <summary>
        /// Tries to get the string representation corresponding to the specified type.
        /// </summary>
        /// <param name="type"></param>
        /// <param name="stringValue"></param>
        /// <returns></returns>
        public static bool TryGetStringFromType(Type type, out string stringValue) => _lookup.TryGetValue(_lookup.First(x => x.Value.Type == type).Key, out (Type Type, string StringValue) value) ? (stringValue = value.StringValue) != null : (stringValue = null) != null;

        /// <summary>
        /// Returns the string representation corresponding to the specified type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string GetStringFromType(Type type) => _lookup.First(x => x.Value.Type == type).Value.StringValue;

        /// <summary>
        /// Tries to get the tensor data type corresponding to the specified string representation.
        /// </summary>
        /// <param name="stringValue"></param>
        /// <param name="scalarType"></param>
        /// <returns></returns>
        public static bool TryGetScalarTypeFromString(string stringValue, out ScalarType scalarType) => _lookup.ContainsKey(scalarType = _lookup.First(x => x.Value.StringValue == stringValue).Key);

        /// <summary>
        /// Returns the tensor data type corresponding to the specified string representation.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ScalarType GetScalarTypeFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Tries to get the tensor data type corresponding to the specified type.
        /// </summary>
        /// <param name="type"></param>
        /// <param name="scalarType"></param>
        /// <returns></returns>
        public static bool TryGetScalarTypeFromType(Type type, out ScalarType scalarType) => _lookup.ContainsKey(scalarType = _lookup.First(x => x.Value.Type == type).Key);

        /// <summary>
        /// Returns the tensor data type corresponding to the specified type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static ScalarType GetScalarTypeFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;

        /// <summary>
        /// Tries to get the type corresponding to the specified string representation.
        /// </summary>
        /// <param name="stringValue"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public static bool TryGetTypeFromString(string stringValue, out Type type) => _lookup.TryGetValue(_lookup.First(x => x.Value.StringValue == stringValue).Key, out (Type Type, string StringValue) value) ? (type = value.Type) != null : (type = null) != null;

        /// <summary>
        /// Returns the type corresponding to the specified string representation.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static Type GetTypeFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Value.Type;
    }
}