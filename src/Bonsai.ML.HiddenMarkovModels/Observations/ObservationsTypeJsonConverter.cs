using Newtonsoft.Json;
using System;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="ObservationsType"/> and the corresponding Python string representation.
    /// </summary>
    public class ObservationsTypeJsonConverter : JsonConverter<ObservationsType>
    {
        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, ObservationsType value, JsonSerializer serializer)
        {
            writer.WriteValue(GetString(value));
        }

        /// <inheritdoc/>
        public override ObservationsType ReadJson(JsonReader reader, Type objectType, ObservationsType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return GetFromString(stringValue);
        }
    }
}