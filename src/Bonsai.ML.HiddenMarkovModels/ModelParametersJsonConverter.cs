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

            result.ObservationsModelType = result.StateParameters?.Observations?.ObservationsModelType 
                ?? ObservationsModelLookup.GetFromString(jo["observations"]?.ToObject<string>());

            result.TransitionsModelType = result.StateParameters?.Transitions?.TransitionsModelType 
                ?? TransitionsModelLookup.GetFromString(jo["transitions"]?.ToObject<string>());

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
                    writer.WritePropertyName("observations");
                    serializer.Serialize(writer, value.ObservationsModelType);
                }

                if (value.StateParameters.Transitions == null)
                {
                    writer.WritePropertyName("transitions");
                    serializer.Serialize(writer, value.TransitionsModelType);
                }
            }
            else
            {
                writer.WritePropertyName("observations");
                serializer.Serialize(writer, ObservationsModelLookup.GetString(value.ObservationsModelType));

                writer.WritePropertyName("transitions");
                serializer.Serialize(writer, TransitionsModelLookup.GetString(value.TransitionsModelType));
            }

            writer.WriteEndObject();
        }
    }
}