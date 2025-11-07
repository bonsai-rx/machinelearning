using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using Bonsai.ML.Hmm.Python.Observations;
using Bonsai.ML.Hmm.Python.Transitions;

namespace Bonsai.ML.Hmm.Python
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
            ModelParameters result = new();

            result.NumStates = jo["num_states"]?.ToObject<int>() ?? result.NumStates;
            result.Dimensions = jo["dimensions"]?.ToObject<int>() ?? result.Dimensions;
            result.StateParameters = jo["StateParameters"]?.ToObject<StateParameters>();

            result.ObservationModelType = result.StateParameters?.Observations?.ObservationModelType 
                ?? ObservationModelLookup.GetFromString(jo["observation_model_type"]?.ToObject<string>());

            result.TransitionModelType = result.StateParameters?.Transitions?.TransitionModelType 
                ?? TransitionModelLookup.GetFromString(jo["transition_model_type"]?.ToObject<string>());

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
                    writer.WritePropertyName("observation_model_type");
                    serializer.Serialize(writer, value.ObservationModelType);
                }

                if (value.StateParameters.Transitions == null)
                {
                    writer.WritePropertyName("transition_model_type");
                    serializer.Serialize(writer, value.TransitionModelType);
                }
            }
            else
            {
                writer.WritePropertyName("observation_model_type");
                serializer.Serialize(writer, ObservationModelLookup.GetString(value.ObservationModelType));

                writer.WritePropertyName("transition_model_type");
                serializer.Serialize(writer, TransitionModelLookup.GetString(value.TransitionModelType));
            }

            writer.WriteEndObject();
        }
    }
}