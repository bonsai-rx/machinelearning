using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;

namespace Bonsai.ML.Torch.LinearAlgebra;

/// <summary>
/// Represents an operator that computes a vector or matrix norm.
/// </summary>
[Combinator]
[Description("Computes a vector or matrix norm.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Norm
{
    /// <summary>
    /// Gets or sets the dimensions along which to compute the norm.
    /// </summary>
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    [Description("The dimensions along which to compute the norm.")]
    public long[]? Dimensions { get; set; } = null;

    /// <summary>
    /// Gets or sets whether the reduced dimensions are retained in the result as dimensions with size one.
    /// </summary>
    /// <remarks>
    /// True indicates that the reduced dimensions are retained in the result as dimensions with size one; otherwise, the reduced dimensions are removed from the result.
    /// </remarks>
    [Description("True indicates that the reduced dimensions are retained in the result as dimensions with size one; otherwise, the reduced dimensions are removed from the result.")]
    public bool Keepdim { get; set; }

    /// <summary>
    /// Computes a matrix norm.
    /// </summary>
    public IObservable<Tensor> Process(IObservable<Tensor> source)
    {
        return source.Select(tensor => linalg.norm(tensor, dims: Dimensions, keepdim: Keepdim));
    }
}
