using System.Linq;
using System.ComponentModel;
using System.Reactive.Linq;
using System;

namespace Bonsai.ML.Python
{
    /// <summary>
    /// Represents an operator that can convert an object into a properly formatted string that is consistent with python syntax.
    /// For example, a tuple (1, 2, 3) will be converted to the string "(1, 2, 3)". A list of [0, 1, 2] will be converted to the string "[0, 1, 2]".
    /// </summary>
    [Combinator]
    [Description("Represents an operator that can convert an object into a properly formatted string that is consistent with python.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class FormatToPython
    {
        /// <summary>
        /// Transforms the elements of an observable sequence into a properly formatted string that is consistent with python syntax.
        /// </summary>
        public IObservable<string> Process<TSource>(IObservable<TSource> source)
        {
            var stringFormatter = new StringFormatter();
            return source.Select(value => {
                return stringFormatter.Format(value);
            });
        }
    }
}
