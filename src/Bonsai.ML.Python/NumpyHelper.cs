using System;
using Python.Runtime;
using System.Collections.Generic;
using System.ComponentModel;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Text;
using System.Linq;
using System.Runtime.InteropServices;

namespace Bonsai.ML.Python
{
    /// <summary>
    /// Provides a set of static methods for working with NumPy arrays.
    /// </summary>
    public static class NumpyHelper
    {
        /// <summary>
        /// Represents a NumPy array interface for interacting with <see cref="PyObject"/> representing NumPy arrays.
        /// </summary>
        public class NumpyArrayInterface
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NumpyArrayInterface"/> class.
            /// </summary>
            public NumpyArrayInterface(PyObject obj)
            {
                if (!IsNumPyArray(obj))
                {
                    throw new ArgumentException($"Object is not a numpy array.", nameof(obj));
                }
                var meta = obj.GetAttr("__array_interface__");
                IsCStyleContiguous = meta["strides"] == null;
                Address = new IntPtr(meta["data"][0].As<long>());

                var typestr = meta["typestr"].As<string>();
                var dtype = typestr.Substring(1);
                switch (dtype)
                {
                    case "b1":
                        DataType = typeof(bool);
                        break;
                    case "f4":
                        DataType = typeof(float);
                        break;
                    case "f8":
                        DataType = typeof(double);
                        break;
                    case "i2":
                        DataType = typeof(short);
                        break;
                    case "i4":
                        DataType = typeof(int);
                        break;
                    case "i8":
                        DataType = typeof(long);
                        break;
                    case "u1":
                        DataType = typeof(byte);
                        break;
                    case "u2":
                        DataType = typeof(ushort);
                        break;
                    case "u4":
                        DataType = typeof(uint);
                        break;
                    case "u8":
                        DataType = typeof(ulong);
                        break;
                    default:
                        throw new ArgumentException($"Type is not currently supported.", nameof(dtype));
                }
                Shape = obj.GetAttr("shape").As<long[]>();
                NBytes = obj.GetAttr("nbytes").As<int>();
            }

            /// <summary>
            /// The memory address of the NumPy array data.
            /// </summary>
            public readonly IntPtr Address;

            /// <summary>
            /// The C# data type representing the elements of the NumPy array.
            /// </summary>
            public readonly Type DataType;

            /// <summary>
            /// The shape of the NumPy array.
            /// </summary>
            public readonly long[] Shape;

            /// <summary>
            /// The number of bytes in the NumPy array.
            /// </summary>
            public readonly int NBytes;

            /// <summary>
            /// A value indicating whether the NumPy array is C-style contiguous.
            /// </summary>
            public readonly bool IsCStyleContiguous;
        }

        /// <summary>
        /// Converts a <see cref="PyObject"/> representing a NumPy array to a C# <see cref="Array"/>.
        /// </summary>
        public static Array PyObjectToArray(PyObject array)
        {
            var info = new NumpyArrayInterface(array);
            byte[] data = new byte[info.NBytes];
            Marshal.Copy(info.Address, data, 0, info.NBytes);
            if (info.DataType == typeof(byte) && info.Shape.Length == 1)
            {
                return data;
            }
            var result = Array.CreateInstance(info.DataType, info.Shape);
            Buffer.BlockCopy(data, 0, result, 0, info.NBytes);
            return result;
        }

        private static PyObject deepcopy;

        private static readonly Lazy<PyObject> np = new(InitializeNumpy);

        private static readonly Dictionary<Type, PyObject> np_dtypes = new();

        private static readonly Dictionary<string, Type> csharp_dtypes = new()
        {
            { "uint8",      typeof(byte)    },
            { "uint16",     typeof(ushort)  },
            { "uint32",     typeof(uint)    },
            { "uint64",     typeof(ulong)   },
            { "int16",      typeof(short)   },
            { "int32",      typeof(int)     },
            { "int64",      typeof(long)    },
            { "float32",    typeof(float)   },
            { "float64",    typeof(double)  },
        };

        /// <summary>
        /// Initializes the NumPy module and returns a reference to the module.
        /// </summary>
        public static PyObject InitializeNumpy()
        {
            var np = Py.Import("numpy");
            np_dtypes.Add(typeof(byte), np.GetAttr("uint8"));
            np_dtypes.Add(typeof(ushort), np.GetAttr("uint16"));
            np_dtypes.Add(typeof(uint), np.GetAttr("uint32"));
            np_dtypes.Add(typeof(ulong), np.GetAttr("uint64"));
            np_dtypes.Add(typeof(short), np.GetAttr("int16"));
            np_dtypes.Add(typeof(int), np.GetAttr("int32"));
            np_dtypes.Add(typeof(long), np.GetAttr("int64"));
            np_dtypes.Add(typeof(float), np.GetAttr("float32"));
            np_dtypes.Add(typeof(double), np.GetAttr("float64"));
            var copy = Py.Import("copy");
            deepcopy = copy.GetAttr("deepcopy");
            return np;
        }

        /// <summary>
        /// Checks if the <see cref="PyObject"/> is a type of NumPy array.
        /// </summary>
        public static bool IsNumPyArray(PyObject obj)
        {
            dynamic numpy = np.Value;
            return numpy.ndarray.__instancecheck__(obj);
        }

        /// <summary>
        /// Gets the NumPy data type for the specified C# type.
        /// </summary>
        public static PyObject GetNumpyDataType(Type type)
        {
            PyObject dtype;
            np_dtypes.TryGetValue(type, out dtype);
            if (dtype == null)
            {
                throw new ArgumentException("Type is not currently supported.", nameof(type));
            }
            return dtype;
        }

        /// <summary>
        /// Gets the C# data type for the specified NumPy data type.
        /// </summary>
        public static Type GetCSharpDataType(string str)
        {
            Type type;
            csharp_dtypes.TryGetValue(str, out type);
            if (type == null)
            {
                throw new ArgumentException("Could not determine data type from string. Data type is either incorrect or not supported.", nameof(str));
            }
            return type;
        }

        /// <summary>
        /// A custom type converter for NumPy data types.
        /// </summary>
        public class NumpyDataTypes : StringConverter
        {
            /// <inheritdoc/>
            public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
            {
                return true;
            }

            /// <inheritdoc/>
            public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
            {
                var dtypes = new List<string>
                {
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "float32",
                    "float64"
                };
                return new StandardValuesCollection(dtypes);
            }
        }
    }
}
