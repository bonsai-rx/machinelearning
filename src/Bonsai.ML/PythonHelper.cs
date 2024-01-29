using Python.Runtime;
using System;
using System.Collections.Generic;

namespace Bonsai.ML
{
    public class PythonHelper
    {
        public static object GetPythonAttribute(PyObject pyObject, string attributeName)
        {
            using (Py.GIL())
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
                    throw new InvalidOperationException($"Attribute {attributeName} not found in Python object.");
                }
            }
        }

        public static T GetPythonAttribute<T>(PyObject pyObject, string attributeName)
        {
            using (Py.GIL())
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
                    throw new InvalidOperationException($"Attribute {attributeName} not found in Python object.");
                }
            }
        }

        public static object ConvertPythonObjectToCSharp(PyObject pyObject)
        {
            if (PyInt.IsIntType(pyObject))
            {
                return pyObject.As<int>();
            }
            else if (PyFloat.IsFloatType(pyObject))
            {
                return pyObject.As<double>();
            }
            else if (PyString.IsStringType(pyObject))
            {
                return pyObject.As<string>();
            }

            else if (PyList.IsListType(pyObject))
            {
                var pyList = new PyList(pyObject);
                var resultList = new List<object>();
                foreach (PyObject item in pyList)
                    resultList.Add(ConvertPythonObjectToCSharp(item));
                return resultList;
            }

            else if (PyDict.IsDictType(pyObject))
            {
                var pyDict = new PyDict(pyObject);
                var resultDict = new Dictionary<object, object>();
                foreach (PyObject key in pyDict.Keys())
                {
                    var value = pyDict[key];
                    resultDict.Add(ConvertPythonObjectToCSharp(key), ConvertPythonObjectToCSharp(value));
                }
                return resultDict;
            }

            else if (IsNumPyArray(pyObject))
            {
                return ConvertNumPyArrayToList(pyObject);
            }

            throw new InvalidOperationException($"Unable to convert python data type to C#. Allowed data types include: integer, float, string, list, dictionary, and numpy arrays");
        }

        public static bool IsNumPyArray(PyObject obj)
        {
            dynamic np = Py.Import("numpy");
            return np.ndarray.__instancecheck__(obj);
        }

        public static object ConvertNumPyArrayToList(dynamic npArray)
        {
            var shape = npArray.shape;
            long dimensions = shape.Length();

            if (dimensions == 0)             
            {
                return new List<object>(ConvertPythonObjectToCSharp(npArray));
            }

            int length = shape[0];
            var resultList = new List<object>(length);

            if (dimensions == 1)
            {
                for (int i = 0; i < length; i++)
                {
                    resultList.Add(ConvertPythonObjectToCSharp(npArray[i]));
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    resultList.Add(ConvertNumPyArrayToList(npArray[i]));
                }
            }

            return resultList;
        }
    }
}