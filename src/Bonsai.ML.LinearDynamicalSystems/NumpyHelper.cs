using System;
using Python.Runtime;
using System.Collections.Generic;
using System.ComponentModel;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Text;
using System.Linq;
using System.Runtime.InteropServices;

namespace Bonsai.ML.LinearDynamicalSystems
{
    static class NumpyHelper
    {
        public class NumpyArrayInterface
        {
            public NumpyArrayInterface(PyObject obj)
            {
                if (!IsNumPyArray(obj))
                {
                    throw new Exception("object is not a numpy array");
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
                        throw new Exception($"Type '{dtype}' not supported");
                }
                Shape = obj.GetAttr("shape").As<long[]>();
                NBytes = obj.GetAttr("nbytes").As<int>();
            }

            public readonly IntPtr Address;

            public readonly Type DataType;

            public readonly long[] Shape;

            public readonly int NBytes;

            public readonly bool IsCStyleContiguous;
        }

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

        private static readonly Lazy<PyObject> np = new Lazy<PyObject>(InitializeNumpy);

        private static readonly Dictionary<Type, PyObject> np_dtypes = new Dictionary<Type, PyObject>();

        private static readonly Dictionary<string, Type> csharp_dtypes = new Dictionary<string, Type>(){
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

        public static bool IsNumPyArray(PyObject obj)
        {
            dynamic numpy = np.Value;
            return numpy.ndarray.__instancecheck__(obj);
        }

        public static PyObject GetNumpyDataType(Type type)
        {
            PyObject dtype;
            np_dtypes.TryGetValue(type, out dtype);
            if (dtype == null)
            {
                throw new Exception($"type '{type}' not supported.");
            }
            return dtype;
        }

        public static Type GetCSharpDataType(string str)
        {
            Type type;
            csharp_dtypes.TryGetValue(str, out type);
            if (type == null)
            {
                throw new Exception($"type '{type}' not supported.");
            }
            return type;
        }

        public class NumpyDataTypes : StringConverter
        {
            public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
            {
                return true;
            }

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

        public class NumpyParser
        {
            public static string ParseArray(Array array)
            {
                StringBuilder sb = new StringBuilder();
                ParseArrayToStringRecursive(array, sb, new int[0]);
                return sb.ToString();
            }

            private static void ParseArrayToStringRecursive(Array array, StringBuilder sb, int[] indices)
            {
                if (indices.Length < array.Rank)
                {
                    sb.Append("[");
                    int length = array.GetLength(indices.Length);
                    for (int i = 0; i < length; i++)
                    {
                        int[] newIndices = new int[indices.Length + 1];
                        indices.CopyTo(newIndices, 0);
                        newIndices[indices.Length] = i;
                        ParseArrayToStringRecursive(array, sb, newIndices);
                        if (i < length - 1)
                        {
                            sb.Append(", ");
                        }
                    }
                    sb.Append("]");
                }
                else
                {
                    object value = array.GetValue(indices);
                    sb.Append(value.ToString());
                }
            }

            private static bool IsValidJson(string input)
            {
                int squareBrackets = 0;
                foreach (char c in input)
                {
                    if (c == '[') squareBrackets++;
                    else if (c == ']') squareBrackets--;
                }
                return squareBrackets == 0;
            }

            public static Array ParseString(string input, Type dtype)
            {
                try
                {
                    if (!IsValidJson(input))
                    {
                        throw new ArgumentException("JSON is invalid.");
                    }
                    var obj = JsonConvert.DeserializeObject<JToken>(input);
                    int depth = ParseDepth(obj);
                    int[] dimensions = ParseDimensions(obj, depth);
                    var resultArray = Array.CreateInstance(dtype, dimensions);
                    PopulateArray(obj, resultArray, new int[0], dtype);
                    return resultArray;
                }
                catch (Exception ex)
                {
                    throw new Exception("Error parsing input string.", ex);
                }
            }

            private static int ParseDepth(JToken token, int currentDepth = 0)
            {
                if (token is JArray arr && arr.Count > 0)
                {
                    return ParseDepth(arr[0], currentDepth + 1);
                }
                return currentDepth;
            }

            private static int[] ParseDimensions(JToken token, int depth, int currentLevel = 0)
            {
                if (depth == 0 || !(token is JArray))
                {
                    return new int[0];
                }

                List<int> dimensions = new List<int>();
                JToken current = token;

                while (current != null && current is JArray)
                {
                    JArray currentArray = current as JArray;
                    dimensions.Add(currentArray.Count);
                    if (currentArray.Count > 0)
                    {
                        if (currentArray.Any(item => !(item is JArray)) && currentArray.Any(item => item is JArray) || currentArray.All(item => item is JArray) && currentArray.Any(item => ((JArray)item).Count != ((JArray)currentArray.First()).Count))
                        {
                            throw new Exception("Error parsing input string.");
                        }

                        if (!(currentArray.First() is JArray) && !currentArray.All(item => double.TryParse(item.ToString(), out _)))
                        {
                            throw new Exception("Error parsing non numeric types.");
                        }
                    }

                    current = currentArray.Count > 0 ? currentArray[0] : null;
                }

                if (currentLevel > 0 && token is JArray arr && arr.All(x => x is JArray))
                {
                    var subArrayDimensions = new HashSet<string>();
                    foreach (JArray subArr in arr)
                    {
                        int[] subDims = ParseDimensions(subArr, depth - currentLevel, currentLevel + 1);
                        subArrayDimensions.Add(string.Join(",", subDims));
                    }

                    if (subArrayDimensions.Count > 1)
                    {
                        throw new ArgumentException("Inconsistent array dimensions.");
                    }
                }

                return dimensions.ToArray();
            }

            private static void PopulateArray(JToken token, Array array, int[] indices, Type dtype)
            {
                if (token is JArray arr)
                {
                    for (int i = 0; i < arr.Count; i++)
                    {
                        int[] newIndices = new int[indices.Length + 1];
                        Array.Copy(indices, newIndices, indices.Length);
                        newIndices[newIndices.Length - 1] = i;
                        PopulateArray(arr[i], array, newIndices, dtype);
                    }
                }
                else
                {
                    var values = ParseType(token, dtype);
                    array.SetValue(values, indices);
                }
            }

            private static object ParseType(object value, Type targetType)
            {
                try
                {
                    return Convert.ChangeType(value, targetType);
                }
                catch (Exception ex)
                {
                    throw new Exception("Error parsing type: ", ex);
                }
            }
        }
    }
}
