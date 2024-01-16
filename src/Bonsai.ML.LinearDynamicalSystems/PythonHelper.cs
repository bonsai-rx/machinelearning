namespace Bonsai.ML.LinearDynamicalSystems
{
    public class PythonHelper
    {
        public static object GetPythonAttribute(Python.Runtime.PyObject pyObject, string attributeName)
        {
            if (pyObject.HasAttr(attributeName))
            {
                using (var attr = pyObject.GetAttr(attributeName))
                {
                    return ConvertPythonObjectToCSharp(attr);
                }
            }
            else
            {
                throw new System.InvalidOperationException($"Attribute {attributeName} not found in Python object.");
            }
        }

        public static T GetPythonAttribute<T>(Python.Runtime.PyObject pyObject, string attributeName)
        {
            if (pyObject.HasAttr(attributeName))
            {
                using (var attr = pyObject.GetAttr(attributeName))
                {
                    return (T)attr.AsManagedObject(typeof(T));
                }
            }
            else
            {
                throw new System.InvalidOperationException($"Attribute {attributeName} not found in Python object.");
            }
        }

        public static object ConvertPythonObjectToCSharp(Python.Runtime.PyObject pyObject)
        {
            // Handle basic types (int, float, string)
            if (Python.Runtime.PyInt.IsIntType(pyObject))
            {
                return pyObject.As<int>();
            }
            else if (Python.Runtime.PyFloat.IsFloatType(pyObject))
            {
                return pyObject.As<double>();
            }
            else if (Python.Runtime.PyString.IsStringType(pyObject))
            {
                return pyObject.As<string>();
            }

            // Handle list to List<object>
            else if (Python.Runtime.PyList.IsListType(pyObject))
            {
                var pyList = new Python.Runtime.PyList(pyObject);
                var resultList = new System.Collections.Generic.List<object>();
                foreach (Python.Runtime.PyObject item in pyList)
                    resultList.Add(ConvertPythonObjectToCSharp(item));
                return resultList;
            }

            // Handle dictionary to Dictionary<object, object>
            else if (Python.Runtime.PyDict.IsDictType(pyObject))
            {
                var pyDict = new Python.Runtime.PyDict(pyObject);
                var resultDict = new System.Collections.Generic.Dictionary<object, object>();
                foreach (Python.Runtime.PyObject key in pyDict.Keys())
                {
                    var value = pyDict[key];
                    resultDict.Add(ConvertPythonObjectToCSharp(key), ConvertPythonObjectToCSharp(value));
                }
                return resultDict;
            }

            // Handle NumPy array to List<object>
            else if (IsNumPyArray(pyObject))
            {
                return ConvertNumPyArrayToList(pyObject);
            }

            // Other types
            // You can add more conversions or return the PyObject directly
            throw new System.InvalidOperationException($"Unable to convert python data type to C#. Allowed data types include: integer, float, string, list, dictionary, and numpy arrays");
        }

        public static bool IsNumPyArray(Python.Runtime.PyObject obj)
        {
            dynamic np = Python.Runtime.Py.Import("numpy");
            return np.ndarray.__instancecheck__(obj);
        }

        public static object ConvertNumPyArrayToList(dynamic npArray)
        {
            var shape = npArray.shape;
            long dimensions = shape.Length();

            if (dimensions == 0) // Scalar
            {
                return new System.Collections.Generic.List<object>(ConvertPythonObjectToCSharp(npArray));
            }

            int length = shape[0];
            var resultList = new System.Collections.Generic.List<object>(length);

            if (dimensions == 1) // 1D Array
            {
                for (int i = 0; i < length; i++)
                {
                    resultList.Add(ConvertPythonObjectToCSharp(npArray[i]));
                }
            }
            else // Multi-dimensional array
            {
                for (int i = 0; i < length; i++)
                {
                    resultList.Add(ConvertNumPyArrayToList(npArray[i]));
                }
            }

            return resultList;
        }
        
        static readonly System.Collections.Generic.Dictionary<string, string> PrimitiveTypes = new()
        {
            { "bool",    "System.Boolean" },
            { "byte",    "System.Byte" },
            { "sbyte",   "System.SByte" },
            { "char",    "System.Char" },
            { "decimal", "System.Decimal" },
            { "double",  "System.Double" },
            { "float",   "System.Single" },
            { "int",     "System.Int32" },
            { "uint",    "System.UInt32" },
            { "nint",    "System.IntPtr" },
            { "nuint",   "System.UIntPtr" },
            { "long",    "System.Int64" },
            { "ulong",   "System.UInt64" },
            { "short",   "System.Int16" },
            { "ushort",  "System.UInt16" },
            { "string",  "System.String" }
        };
    }

}