using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.Tensors
{
    /// <summary>
    /// Converts the input tensor into an OpenCV mat.
    /// </summary>
    [Combinator]
    [Description("Converts the input tensor into an OpenCV mat.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ToMat
    {
        /// <summary>
        /// Converts the input tensor into an OpenCV mat.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<Mat> Process(IObservable<Tensor> source)
        {
            return source.Select(Helpers.OpenCVHelper.ToMat);
        }
    }
}