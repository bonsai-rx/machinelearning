using Newtonsoft.Json;
using System;

namespace Bonsai.ML.Hmm.Python.Transitions
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="TransitionModelType"/> and the corresponding Python string representation.
    /// </summary>
    public class TransitionModelTypeJsonConverter : JsonConverter<TransitionModelType>
    {
        /// <inheritdoc/>
        public override TransitionModelType ReadJson(JsonReader reader, Type objectType, TransitionModelType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return TransitionModelLookup.GetFromString(stringValue);
        }

        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, TransitionModelType value, JsonSerializer serializer)
        {
            writer.WriteValue(TransitionModelLookup.GetString(value));
        }
    }
}