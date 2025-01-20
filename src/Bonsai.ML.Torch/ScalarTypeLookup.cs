using System;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Provides methods to look up tensor data types.
    /// </summary>
    public static class ScalarTypeLookup
    {
        private static readonly Dictionary<ScalarType, (Type Type, string StringValue)> _lookup = new()
        {
            { ScalarType.Byte, (typeof(byte), "byte") },
            { ScalarType.Int16, (typeof(short), "short") },
            { ScalarType.Int32, (typeof(int), "int") },
            { ScalarType.Int64, (typeof(long), "long") },
            { ScalarType.Float32, (typeof(float), "float") },
            { ScalarType.Float64, (typeof(double), "double") },
            { ScalarType.Bool, (typeof(bool), "bool") },
            { ScalarType.Int8, (typeof(sbyte), "sbyte") },
        };

        /// <summary>
        /// Returns the type corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static Type GetTypeFromScalarType(ScalarType type) => _lookup[type].Type;

        /// <summary>
        /// Returns the string representation corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string GetStringFromScalarType(ScalarType type) => _lookup[type].StringValue;

        /// <summary>
        /// Returns the tensor data type corresponding to the specified string representation.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static ScalarType GetScalarTypeFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Returns the tensor data type corresponding to the specified type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static ScalarType GetScalarTypeFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}