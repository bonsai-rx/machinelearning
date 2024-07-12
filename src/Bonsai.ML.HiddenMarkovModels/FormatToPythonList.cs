using System.Collections;
using System.Linq;
using System.Text;
using System.ComponentModel;
using System.Reactive.Linq;
using System;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Represents an operator that can convert an object into a properly formatted string that is consistent with a python list.
    /// For example, a tuple (1, 2, 3) will be converted to the string "[1, 2, 3]".
    /// Does not support nested/multidimensional enumerables, e.g. a tuple of tuples, a list of lists, etc.
    /// </summary>
    [Combinator]
    [Description("Converts an object into a properly formatted string that is consistent with a python list.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class FormatToPythonList
    {
        /// <summary>
        /// Transforms the elements of an observable sequence into a properly formatted string that is consistent with a python list.
        /// </summary>
        public IObservable<string> Process<TSource>(IObservable<TSource> source)
        {
            return source.Select(value => PythonListHelper.ConvertToPythonListString(value));
        }
    }

    internal static class PythonListHelper
    {
        public static string ConvertToPythonListString(object obj)
        {
            return $"[{ConvertToCommaSeparatedString(obj)}]";
        }

        public static string ConvertToCommaSeparatedString(object obj)
        {
            if (obj == null) return "None";

            if (obj is string)
            {
                return obj.ToString();
            }

            if (obj is IEnumerable enumerable)
            {
                var sb = new StringBuilder();
                foreach (var item in enumerable)
                {
                    if (sb.Length > 0)
                    {
                        sb.Append(", ");
                    }
                    sb.Append(ConvertToCommaSeparatedString(item));
                }
                return sb.ToString();
            }

            var tupleType = obj.GetType();
            if (tupleType.IsGenericType && tupleType.FullName.StartsWith("System.Tuple"))
            {
                var values = tupleType.GetProperties()
                                    .Where(p => p.Name.StartsWith("Item"))
                                    .Select(p => p.GetValue(obj));

                var sb = new StringBuilder();
                foreach (var value in values)
                {
                    if (sb.Length > 0)
                    {
                        sb.Append(", ");
                    }
                    sb.Append(ConvertToCommaSeparatedString(value));
                }
                return sb.ToString();
            }

            if (tupleType.IsValueType && tupleType.FullName.StartsWith("System.ValueTuple"))
            {
                var fields = tupleType.GetFields()
                                    .Where(f => f.Name.StartsWith("Item"))
                                    .Select(f => f.GetValue(obj));

                var sb = new StringBuilder();
                foreach (var field in fields)
                {
                    if (sb.Length > 0)
                    {
                        sb.Append(", ");
                    }
                    sb.Append(ConvertToCommaSeparatedString(field));
                }
                return sb.ToString();
            }

            return obj.ToString();
        }
    }
}
