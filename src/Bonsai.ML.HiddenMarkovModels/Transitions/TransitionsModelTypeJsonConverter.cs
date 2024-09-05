using Newtonsoft.Json;
using System;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="TransitionsModelType"/> and the corresponding Python string representation.
    /// </summary>
    public class TransitionsModelTypeJsonConverter : JsonConverter<TransitionsModelType>
    {
        /// <inheritdoc/>
        public override TransitionsModelType ReadJson(JsonReader reader, Type objectType, TransitionsModelType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return TransitionsModelLookup.GetFromString(stringValue);
        }

        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, TransitionsModelType value, JsonSerializer serializer)
        {
            writer.WriteValue(TransitionsModelLookup.GetString(value));
        }
    }
}