using Newtonsoft.Json;
using System;

namespace Bonsai.ML.Hmm.Python.Observations
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="ObservationModelType"/> and the corresponding Python string representation.
    /// </summary>
    public class ObservationModelTypeJsonConverter : JsonConverter<ObservationModelType>
    {
        /// <inheritdoc/>
        public override ObservationModelType ReadJson(JsonReader reader, Type objectType, ObservationModelType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return ObservationModelLookup.GetFromString(stringValue);
        }

        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, ObservationModelType value, JsonSerializer serializer)
        {
            writer.WriteValue(ObservationModelLookup.GetString(value));
        }
    }
}