using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using Bonsai.ML.Python;

namespace Bonsai.ML.NeuralDecoding;

/// <summary>
/// Transforms the input sequence of Python objects into a sequence of <see cref="PositionRange"/> instances.
/// </summary>
[Combinator]
[Description("Transforms the input sequence of Python objects into a sequence of PositionRange instances.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class PositionRange
{
    /// <summary>
    /// The position bins.
    /// </summary>
    public double[] PositionBins { get; set; }

    /// <summary>
    /// The bin size.
    /// </summary>
    public double BinSize { get; set; }

    /// <summary>
    /// The number of bins.
    /// </summary>
    public int NumberOfBins { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="PositionRange"/> class.
    /// </summary>
    public PositionRange()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PositionRange"/> class.
    /// </summary>
    /// <param name="positionBins"></param>
    public PositionRange(double[] positionBins)
    {
        PositionBins = positionBins;
        BinSize = positionBins[1] - positionBins[0];
        NumberOfBins = positionBins.Length;
    }

    /// <summary>
    /// Transforms the input sequence of Python objects into a sequence of <see cref="PositionRange"/> instances.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<PositionRange> Process(IObservable<PyObject> source)
    {
        return source.Select(value => {
            return new PositionRange((double[])PythonHelper.ConvertPythonObjectToCSharp(value));
        });
    }
}
