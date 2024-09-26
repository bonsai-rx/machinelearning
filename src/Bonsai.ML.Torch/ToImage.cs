using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch
{
    /// <summary>
    /// Converts the input tensor into an OpenCV image.
    /// </summary>
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ToImage
    {
        /// <summary>
        /// Converts the input tensor into an OpenCV image.
        /// </summary>
        /// <param name="source"></param>
        /// <returns></returns>
        public IObservable<IplImage> Process(IObservable<Tensor> source)
        {
            return source.Select(OpenCVHelper.ToImage);
        }
    }
}