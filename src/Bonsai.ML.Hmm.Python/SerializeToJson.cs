using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Newtonsoft.Json;

namespace Bonsai.ML.Hmm.Python
{
    /// <summary>
    /// Serializes a sequence of data model objects into JSON strings.
    /// </summary>
    [Combinator]
    [WorkflowElementCategory(ElementCategory.Transform)]
    [Description("Serializes a sequence of data model objects into JSON strings.")]
    public class SerializeToJson
    {
        private IObservable<string> Process<T>(IObservable<T> source)
        {
            return source.Select(value => JsonConvert.SerializeObject(value));
        }

        /// <summary>
        /// Serializes each <see cref="StateParameters"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="StateParameters"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="StateParameters"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<StateParameters> source)
        {
            return Process<StateParameters>(source);
        }

        /// <summary>
        /// Serializes each <see cref="ModelParameters"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="ModelParameters"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="ModelParameters"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<ModelParameters> source)
        {
            return Process<ModelParameters>(source);
        }
    }
}
