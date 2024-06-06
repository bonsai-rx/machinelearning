using System;
using System.ComponentModel;
using System.Reactive.Linq;
using Newtonsoft.Json;

namespace Bonsai.ML.HiddenMarkovModels
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
        /// Serializes each <see cref="State"/> object in the sequence to
        /// a JSON string.
        /// </summary>
        /// <param name="source">
        /// A sequence of <see cref="State"/> objects.
        /// </param>
        /// <returns>
        /// A sequence of JSON strings representing the corresponding
        /// <see cref="State"/> object.
        /// </returns>
        public IObservable<string> Process(IObservable<State> source)
        {
            return Process<State>(source);
        }
    }
}
