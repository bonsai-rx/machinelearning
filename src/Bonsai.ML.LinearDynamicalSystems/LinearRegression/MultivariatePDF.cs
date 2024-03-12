using System.ComponentModel;
using System;
using System.Reactive.Linq;
using Python.Runtime;
using System.Xml.Serialization;

namespace Bonsai.ML.LinearDynamicalSystems.LinearRegression
{
    /// <summary>
    /// A class that converts a python object, representing a multivariate PDF, into a multidimensional array
    /// /// </summary>
    [Description("A multivariate PDF")]
    [Combinator()]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class MultivariatePDF
    {

        [XmlIgnore]
        public GridParameters GridParameters;

        [XmlIgnore]
        public double[,] Values;

        /// <summary>
        /// Converts a python object, representing a multivariate PDF, into a multidimensional array
        /// </summary>
        public IObservable<MultivariatePDF> Process(IObservable<PyObject> source)
        {
            return Observable.Select(source, pyObject =>
            {
                var gridParameters = GridParameters.ConvertPyObject(pyObject);
                var values = (double[,])pyObject.GetArrayAttribute("pdf_values");
                return new MultivariatePDF {
                    GridParameters = gridParameters,
                    Values = values
                };
            });
        }
    }
}