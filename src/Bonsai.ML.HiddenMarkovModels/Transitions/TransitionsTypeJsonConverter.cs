using Newtonsoft.Json;
using System;
using static Bonsai.ML.HiddenMarkovModels.Transitions.TransitionsLookup;

namespace Bonsai.ML.HiddenMarkovModels.Transitions
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="TransitionsType"/> and the corresponding Python string representation.
    /// </summary>
    public class TransitionsTypeJsonConverter : JsonConverter<TransitionsType>
    {
        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, TransitionsType value, JsonSerializer serializer)
        {
            writer.WriteValue(GetString(value));
        }

        /// <inheritdoc/>
        public override TransitionsType ReadJson(JsonReader reader, Type objectType, TransitionsType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return GetFromString(stringValue);
        }
    }
}