using Newtonsoft.Json;
using System;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public class ObservationsTypeJsonConverter : JsonConverter<ObservationsType>
    {
        public override void WriteJson(JsonWriter writer, ObservationsType value, JsonSerializer serializer)
        {
            writer.WriteValue(GetString(value));
        }

        public override ObservationsType ReadJson(JsonReader reader, Type objectType, ObservationsType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return GetFromString(stringValue);
        }
    }
}