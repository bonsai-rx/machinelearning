using Newtonsoft.Json;
using System;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="ObservationsModelType"/> and the corresponding Python string representation.
    /// </summary>
    public class ObservationsModelTypeJsonConverter : JsonConverter<ObservationsModelType>
    {
        /// <inheritdoc/>
        public override ObservationsModelType ReadJson(JsonReader reader, Type objectType, ObservationsModelType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return ObservationsModelLookup.GetFromString(stringValue);
        }

        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, ObservationsModelType value, JsonSerializer serializer)
        {
            writer.WriteValue(ObservationsModelLookup.GetString(value));
        }
    }
}