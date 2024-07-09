using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Linq;
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
            var observationsType = GetFromString(jo["Observations"]["ObservationsType"]?.ToString());

            var paramsArray = (JArray)jo["Observations"]["Params"];
            object[] paramsObjArr = new object[paramsArray.Count];
            for (int i = 0; i < paramsArray.Count; i++)
            {
                paramsObjArr[i] = NumpyHelper.NumpyParser.ParseString(paramsArray[i].ToString(), typeof(double));
            }

            var kwargs = (JObject)jo["Observations"]["Kwargs"];
            object[] kwargsArray = kwargs.Properties()
                        .Select(p => p.Value.ToObject<object>())
                        .ToArray();

            var observations = kwargsArray.Count() == 0 
                ? (ObservationsModel)Activator.CreateInstance(GetObservationsClassType(observationsType)) 
                : (ObservationsModel)Activator.CreateInstance(GetObservationsClassType(observationsType), kwargsArray);

            observations.Params = paramsObjArr;
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

            writer.WritePropertyName("Observations");
            serializer.Serialize(writer, value.Observations);

            writer.WriteEndObject();
        }
    }
}
