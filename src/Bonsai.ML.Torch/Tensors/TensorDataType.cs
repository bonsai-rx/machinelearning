using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tensors
{
    /// <summary>
    /// Represents the data type of the tensor elements. Contains currently supported data types. A subset of the available ScalarType data types in TorchSharp.
    /// </summary>
    public enum TensorDataType
    {
        /// <summary>
        /// 8-bit unsigned integer.
        /// </summary>
        Byte = ScalarType.Byte,

        /// <summary>
        /// 8-bit signed integer.
        /// </summary>
        Int8 = ScalarType.Int8,

        /// <summary>
        /// 16-bit signed integer.
        /// </summary>
        Int16 = ScalarType.Int16,

        /// <summary>
        /// 32-bit signed integer.
        /// </summary>
        Int32 = ScalarType.Int32,

        /// <summary>
        /// 64-bit signed integer.
        /// </summary>
        Int64 = ScalarType.Int64,

        /// <summary>
        /// 32-bit floating point.
        /// </summary>
        Float32 = ScalarType.Float32,

        /// <summary>
        /// 64-bit floating point.
        /// </summary>
        Float64 = ScalarType.Float64,

        /// <summary>
        /// Boolean.
        /// </summary>
        Bool = ScalarType.Bool
    }
}