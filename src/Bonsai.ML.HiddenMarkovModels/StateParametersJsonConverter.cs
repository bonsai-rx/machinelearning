using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using Bonsai.ML.HiddenMarkovModels.Observations;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

namespace Bonsai.ML.HiddenMarkovModels
{
    public class StateParametersJsonConverter: JsonConverter<StateParameters>
    {
        public override StateParameters ReadJson(JsonReader reader, Type objectType, StateParameters existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            JObject jo = JObject.Load(reader);
            
            StateParameters result = new StateParameters();

            result.InitialStateDistribution = jo["InitialStateDistribution"]?.ToObject<double[]>();
            result.LogTransitionProbabilities = jo["LogTransitionProbabilities"]?.ToObject<double[,]>();
            result.ObservationType = GetFromString(jo["ObservationType"]?.ToString());

            JArray paramsArray = (JArray)jo["Observations"]["Params"];
            object[] objArr = new object[paramsArray.Count];
            for (int i = 0; i < paramsArray.Count; i++)
            {
                objArr[i] = NumpyHelper.NumpyParser.ParseString(paramsArray[i].ToString(), typeof(double));
            }

            var observations = (ObservationsBase)Activator.CreateInstance(GetObservationsClassType(result.ObservationType));
            observations.Params = objArr;
            result.Observations = observations;
            
            return result;
        }

        public override void WriteJson(JsonWriter writer, StateParameters value, JsonSerializer serializer)
        {
            writer.WriteStartObject();

            writer.WritePropertyName("InitialStateDistribution");
            serializer.Serialize(writer, value.InitialStateDistribution);

            writer.WritePropertyName("LogTransitionProbabilities");
            serializer.Serialize(writer, value.LogTransitionProbabilities);

            writer.WritePropertyName("ObservationType");
            writer.WriteValue(GetString(value.ObservationType));

            writer.WritePropertyName("Observations");
            serializer.Serialize(writer, value.Observations);

            writer.WriteEndObject();
        }
    }
}
