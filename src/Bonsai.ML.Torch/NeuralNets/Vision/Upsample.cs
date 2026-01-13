using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Bonsai.ML.Torch.NeuralNets.Vision;

/// <summary>
/// Creates an Upsample module.
/// </summary>
/// <remarks>
/// See <see href="https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html"/> for more information.
/// </remarks>
[Description("Creates an Upsample module.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class Upsample
{
    /// <summary>
    /// The output spatial sizes.
    /// </summary>
    [Description("The output spatial sizes.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public long[] Size { get; set; } = null;

    /// <summary>
    /// The multiplier for the spatial size.
    /// </summary>
    [Description("The multiplier for the spatial size.")]
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    public double[] ScaleFactor { get; set; } = null;

    /// <summary>
    /// The upsampling algorithm.
    /// </summary>
    [Description("The upsampling algorithm.")]
    public UpsampleMode Mode { get; set; } = UpsampleMode.Nearest;

    /// <summary>
    /// If True, the corner pixels of the input and output tensors are aligned.
    /// </summary>
    /// <remarks>
    /// This only has effect when mode is 'linear', 'bilinear', 'bicubic' or 'trilinear'.
    /// </remarks>
    [Description("If True, the corner pixels of the input and output tensors are aligned.")]
    public bool? AlignCorners { get; set; } = null;

    /// <summary>
    /// Recomputes the scale factor for use in the interpolation calculation.
    /// </summary>
    /// <remarks>
    /// If true, then the scale factor must be passed in and is used to compute the output size.
    /// If false, then size or scale factor will be used for interpolation.
    /// </remarks>
    [Description("Recomputes the scale factor for use in the interpolation calculation.")]
    public bool? RecomputeScaleFactor { get; set; } = null;

    /// <summary>
    /// Creates an Upsample module.
    /// </summary>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process()
    {
        return Observable.Return(Upsample(Size, ScaleFactor, Mode, AlignCorners, RecomputeScaleFactor));
    }

    /// <summary>
    /// Creates an Upsample module.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Module<Tensor, Tensor>> Process<T>(IObservable<T> source)
    {
        return source.Select(_ => Upsample(Size, ScaleFactor, Mode, AlignCorners, RecomputeScaleFactor));
    }
}
