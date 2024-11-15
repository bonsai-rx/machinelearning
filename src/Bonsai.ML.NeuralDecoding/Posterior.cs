using System.ComponentModel;
using System;
using System.Reactive.Linq;
using System.Linq;
using Python.Runtime;
using Bonsai.ML.Python;

namespace Bonsai.ML.NeuralDecoding;

/// <summary>
/// Transforms the input sequence of Python objects into a sequence of <see cref="Posterior"/> instances.
/// </summary>
[Combinator]
[Description("Transforms the input sequence of Python objects into a sequence of Posterior instances.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class Posterior
{
    /// <summary>
    /// The data.
    /// </summary>
    public double[] Data { get; set; }

    /// <summary>
    /// The argmax.
    /// </summary>
    public int ArgMax { get; set; }

    /// <summary>
    /// The value range.
    /// </summary>
    public double[] ValueRange { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="Posterior"/> class.
    /// </summary>
    public Posterior()
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Posterior"/> class.
    /// </summary>
    /// <param name="data"></param>
    /// <param name="argMax"></param>
    /// <param name="valueRange"></param>
    public Posterior(double[] data, int argMax, double[] valueRange)
    {
        Data = data;
        ArgMax = argMax;
        ValueRange = valueRange;
    }

    /// <summary>
    /// Transforms the input sequence of Python objects into a sequence of <see cref="Posterior"/> instances.
    /// </summary>
    /// <param name="source"></param>
    /// <returns></returns>
    public IObservable<Posterior> Process(IObservable<PyObject> source)
    {
        return source.Select(value => {
            var posterior = value[0];
            var valueCenters = value[1];
            var data = (double[])PythonHelper.ConvertPythonObjectToCSharp(posterior);
            var argMax = Array.IndexOf(data, data.Max());
            var valueRange = (double[])PythonHelper.ConvertPythonObjectToCSharp(valueCenters);
            return new Posterior(
                data,
                argMax,
                ValueRange
            );
        });
    }
}