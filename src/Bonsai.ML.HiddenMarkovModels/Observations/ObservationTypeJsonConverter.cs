using Newtonsoft.Json;
using System;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public class ObservationTypeJsonConverter : JsonConverter<ObservationType>
    {
        public override void WriteJson(JsonWriter writer, ObservationType value, JsonSerializer serializer)
        {
            writer.WriteValue(GetString(value));
        }

        public override ObservationType ReadJson(JsonReader reader, Type objectType, ObservationType existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            string stringValue = reader.Value?.ToString();
            return GetFromString(stringValue);
        }
    }
}