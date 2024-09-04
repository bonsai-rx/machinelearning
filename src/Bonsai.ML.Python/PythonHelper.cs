using Python.Runtime;
using System;
using System.Collections.Generic;

namespace Bonsai.ML.Python
{
    /// <summary>
    /// Provides a set of extension methods for working with <see cref="PyObject"/> instances.
    /// </summary>
    public static class PythonHelper
    {
        /// <summary>
        /// Gets the value of the specified array attribute from the Python object.
        /// </summary>
        /// <param name="pyObject">The Python object.</param>
        /// <param name="attributeName">The name of the attribute to retrieve.</param>
        /// <returns>The array value of the specified attribute.</returns>
        public static object GetArrayAttr(this PyObject pyObject, string attributeName)
        {
            using var attr = pyObject.GetAttr(attributeName);
            return ConvertPythonObjectToCSharp(attr);
        }

        /// <summary>
        /// Gets the value of the specified attribute from the Python object.
        /// </summary>
        /// <typeparam name="T">The type of the attribute to retrieve.</typeparam>
        /// <param name="pyObject">The Python object.</param>
        /// <param name="attributeName">The name of the attribute to retrieve.</param>
        /// <returns>The value of the specified attribute.</returns>
        public static T GetAttr<T>(this PyObject pyObject, string attributeName)
        {
            using var attr = pyObject.GetAttr(attributeName);
            if (attr == null || attr.IsNone())
            {
                return default;
            }
            return attr.As<T>();
        }

        /// <summary>
        /// Converts the specified Python object to a C# object.
        /// </summary>
        /// <param name="pyObject">The Python object to convert.</param>
        /// <returns>The C# object representation of the Python object.</returns>
        public static object ConvertPythonObjectToCSharp(PyObject pyObject)
        {
            if (pyObject == null || pyObject.IsNone())
            {
                return null;
            }

            if (PyInt.IsIntType(pyObject))
            {
                return pyObject.As<int>();
            }

            if (PyFloat.IsFloatType(pyObject))
            {
                return pyObject.As<double>();
            }

            if (PyString.IsStringType(pyObject))
            {
                return pyObject.As<string>();
            }

            if (PyList.IsListType(pyObject))
            {
                var pyList = new PyList(pyObject);
                var resultList = new List<object>();
                foreach (PyObject item in pyList)
                    resultList.Add(ConvertPythonObjectToCSharp(item));
                return resultList;
            }

            if (PyDict.IsDictType(pyObject))
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

            if (PyTuple.IsTupleType(pyObject))
            {
                var pyTuple = new PyTuple(pyObject);
                var resultArray = new object[pyTuple.Length()];
                for (int i = 0; i < pyTuple.Length(); i++) {
                    resultArray[i] = ConvertPythonObjectToCSharp(pyTuple[i]);
                }
                return resultArray;
            }

            if (NumpyHelper.IsNumPyArray(pyObject))
            {
                return NumpyHelper.PyObjectToArray(pyObject);
            }

            throw new InvalidOperationException($"Unable to convert python data type to C#. Allowed data types include: integer, float, string, list, dictionary, and numpy arrays. Instead, got: {pyObject.GetPythonType()}");
        }
    }
}
