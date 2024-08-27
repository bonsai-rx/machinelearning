using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tensors
{
    /// <summary>
    /// Converts the input value into a tensor.
    /// </summary>
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ToTensor
    {
        /// <summary>
        /// Converts an int into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<int> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts a double into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<double> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts a byte into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<byte> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts a bool into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<bool> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts a float into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<float> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts a long into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<long> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts a short into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<short> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts an array into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Array> source)
        {
            return source.Select(value => {
                return as_tensor(value);
            });
        }

        /// <summary>
        /// Converts an IplImage into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<IplImage> source)
        {
            return source.Select(Helpers.OpenCVHelper.ToTensor);
        }

        /// <summary>
        /// Converts a Mat into a tensor.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Tensor> Process(IObservable<Mat> source)
        {
            return source.Select(Helpers.OpenCVHelper.ToTensor);
        }
    }
}