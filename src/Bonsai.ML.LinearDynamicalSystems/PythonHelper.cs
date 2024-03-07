using Python.Runtime;
using System;
using System.Collections.Generic;

namespace Bonsai.ML.LinearDynamicalSystems
{
    static class PythonHelper
    {
        public static object GetPythonAttribute(PyObject pyObject, string attributeName)
        {
            using (var attr = pyObject.GetAttr(attributeName))
            {
                return ConvertPythonObjectToCSharp(attr);
            }
        }

        public static T GetPythonAttribute<T>(PyObject pyObject, string attributeName)
        {
            using (var attr = pyObject.GetAttr(attributeName))
            {
                return (T)attr.AsManagedObject(typeof(T));
            }
        }

        public static object ConvertPythonObjectToCSharp(PyObject pyObject)
        {
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

            if (NumpyHelper.IsNumPyArray(pyObject))
            {
                return NumpyHelper.PyObjectToArray(pyObject);
            }

            throw new InvalidOperationException("Unable to convert python data type to C#. Allowed data types include: integer, float, string, list, dictionary, and numpy arrays");
        }
    }
}
