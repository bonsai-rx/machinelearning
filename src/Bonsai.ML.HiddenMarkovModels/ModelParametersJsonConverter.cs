using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using Bonsai.ML.HiddenMarkovModels.Observations;
using Bonsai.ML.HiddenMarkovModels.Transitions;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="ModelParameters"/> and a JSON string representation.
    /// </summary>
    public class ModelParametersJsonConverter : JsonConverter<ModelParameters>
    {
        /// <inheritdoc/>
        public override ModelParameters ReadJson(JsonReader reader, Type objectType, ModelParameters existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            JObject jo = JObject.Load(reader);
            ModelParameters result = new ModelParameters();

            result.NumStates = jo["num_states"]?.ToObject<int>() ?? result.NumStates;
            result.Dimensions = jo["dimensions"]?.ToObject<int>() ?? result.Dimensions;
            result.StateParameters = jo["StateParameters"]?.ToObject<StateParameters>();

            result.ObservationsType = result.StateParameters?.Observations?.ObservationsType 
                ?? ObservationsLookup.GetFromString(jo["observation_type"]?.ToObject<string>());

            result.TransitionsType = result.StateParameters?.Transitions?.TransitionsType 
                ?? TransitionsLookup.GetFromString(jo["transition_type"]?.ToObject<string>());

            return result;
        }

        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, ModelParameters value, JsonSerializer serializer)
        {
            writer.WriteStartObject();

            writer.WritePropertyName("num_states");
            serializer.Serialize(writer, value.NumStates);

            writer.WritePropertyName("dimensions");
            serializer.Serialize(writer, value.Dimensions);

            if (value.StateParameters != null)
            {
                writer.WritePropertyName("StateParameters");
                serializer.Serialize(writer, value.StateParameters);

                if (value.StateParameters.Observations == null)
                {
                    writer.WritePropertyName("observation_type");
                    serializer.Serialize(writer, value.ObservationsType);
                }

                if (value.StateParameters.Transitions == null)
                {
                    writer.WritePropertyName("transitions_type");
                    serializer.Serialize(writer, value.TransitionsType);
                }
            }
            else
            {
                writer.WritePropertyName("observation_type");
                serializer.Serialize(writer, ObservationsLookup.GetString(value.ObservationsType));

                writer.WritePropertyName("transition_type");
                serializer.Serialize(writer, TransitionsLookup.GetString(value.TransitionsType));
            }

            writer.WriteEndObject();
        }
    }
}