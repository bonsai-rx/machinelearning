using Bonsai;
using System;
using System.ComponentModel;
using Python.Runtime;
using System.Reactive.Linq;
using System.Runtime.InteropServices;

namespace Bonsai.ML.Python
{
    [Combinator()]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class NDArray
    {
        private Array content = null;
        private string contentString = null;

        [Editor(DesignTypes.MultilineStringEditor, DesignTypes.UITypeEditor)]
        public string Content
        {
            get => contentString;
            set 
            {
                content = string.IsNullOrEmpty(value) ? null : NumpyHelper.NumpyParser.ParseString(value, type);
                contentString = content != null ? NumpyHelper.NumpyParser.ParseArray(content) : null;
            }
        }

        private Type type = typeof(double);
        private string dtype = "float64";

        [TypeConverter(typeof(NumpyHelper.NumpyDataTypes))]
        public string DType 
        { 
            get => dtype; 
            set 
            {
                if (value != dtype)
                {
                    dtype = value;
                    type = NumpyHelper.GetCSharpDataType(dtype); 
                    content = NumpyHelper.NumpyParser.ParseString(contentString, type);
                    contentString = content != null ? NumpyHelper.NumpyParser.ParseArray(content) : null;
                }
            }
        }

        private PyObject NewArray(Array values)
        {
            NumpyHelper numpyHelper = NumpyHelper.Instance;

            // BlockCopy possibly multidimensional array of arbitrary type to onedimensional byte array
            Type ElementType = values.GetType().GetElementType();
            int nbytes = values.Length * Marshal.SizeOf(ElementType);
            byte[] data = new byte[nbytes];
            Buffer.BlockCopy(values, 0, data, 0, nbytes);

            // Create an python tuple with the dimensions of the input array
            PyObject[] lengths = new PyObject[values.Rank];
            for (int i = 0; i < values.Rank; i++)
                lengths[i] = new PyInt(values.GetLength(i));
            PyTuple shape = new PyTuple(lengths);

            // Create an empty numpy array in correct shape and datatype
            var dtype = NumpyHelper.GetNumpyDataType(type);
            var arr = NumpyHelper.np.InvokeMethod("empty", shape, dtype);

            var meta = arr.GetAttr("__array_interface__");
            var address = new IntPtr(meta["data"][0].As<long>());

            // Copy the data to that array
            Marshal.Copy(data, 0, address, nbytes);

            return arr;
        }

        public IObservable<PyObject> Process()
        {
            if (content == null || content.Length <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(content), "Size of array must be greater than zero.");
            }

            return new GetRuntime().Generate().Select(_ => NewArray(content));
        }
    }
}