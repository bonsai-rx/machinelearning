using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;
using Bonsai.ML.Python;

namespace Bonsai.ML.LinearDynamicalSystems.LinearRegression
{
    /// <summary>
    /// Represents an operator that converts a python object, representing a multivariate PDF, into a multivariate PDF class.
    /// </summary>
    [Combinator]
    [Description("Converts a python object, representing a multivariate PDF, into a multivariate PDF.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class MultivariatePDF
    {

        /// <summary>
        /// Gets or sets the grid parameters used for generating the multivariate PDF.
        /// </summary>
        [XmlIgnore]
        public GridParameters GridParameters;

        /// <summary>
        /// Gets or sets the probability density value at each 2D position of the grid.
        /// </summary>
        [XmlIgnore]
        public double[,] Values;

        /// <summary>
        /// Converts a PyObject into a multivariate PDF.
        /// </summary>
        public IObservable<MultivariatePDF> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var gridParameters = GridParameters.ConvertPyObject(pyObject);
                var values = (double[,])pyObject.GetArrayAttr("pdf_values");
                return new MultivariatePDF {
                    GridParameters = gridParameters,
                    Values = values
                };
            });
        }
    }
}