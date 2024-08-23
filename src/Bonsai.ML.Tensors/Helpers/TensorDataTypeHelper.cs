using System;
using System.Collections.Generic;
using System.Linq;

namespace Bonsai.ML.Tensors.Helpers
{
    /// <summary>
    /// Provides helper methods for working with tensor data types.
    /// </summary>
    public class TensorDataTypeHelper
    {
        private static readonly Dictionary<TensorDataType, (Type Type, string StringValue)> _lookup = new Dictionary<TensorDataType, (Type, string)>
        {
            { TensorDataType.Byte, (typeof(byte), "byte") },
            { TensorDataType.Int16, (typeof(short), "short") },
            { TensorDataType.Int32, (typeof(int), "int") },
            { TensorDataType.Int64, (typeof(long), "long") },
            { TensorDataType.Float32, (typeof(float), "float") },
            { TensorDataType.Float64, (typeof(double), "double") },
            { TensorDataType.Bool, (typeof(bool), "bool") },
            { TensorDataType.Int8, (typeof(sbyte), "sbyte") },
        };

        /// <summary>
        /// Returns the type corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static Type GetTypeFromTensorDataType(TensorDataType type) => _lookup[type].Type;

        /// <summary>
        /// Returns the string representation corresponding to the specified tensor data type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static string GetStringFromTensorDataType(TensorDataType type) => _lookup[type].StringValue;

        /// <summary>
        /// Returns the tensor data type corresponding to the specified string representation.
        /// </summary>
        /// <param name="value"></param>
        /// <returns></returns>
        public static TensorDataType GetTensorDataTypeFromString(string value) => _lookup.First(x => x.Value.StringValue == value).Key;

        /// <summary>
        /// Returns the tensor data type corresponding to the specified type.
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public static TensorDataType GetTensorDataTypeFromType(Type type) => _lookup.First(x => x.Value.Type == type).Key;
    }
}