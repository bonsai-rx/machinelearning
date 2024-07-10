using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Linq;
using Bonsai.ML.HiddenMarkovModels.Observations;
using static Bonsai.ML.HiddenMarkovModels.Observations.ObservationsLookup;

namespace Bonsai.ML.HiddenMarkovModels
{
    /// <summary>
    /// Provides a type converter to convert between <see cref="StateParameters"/> and a JSON string representation.
    /// </summary>
    public class StateParametersJsonConverter : JsonConverter<StateParameters>
    {
        /// <inheritdoc/>
        public override StateParameters ReadJson(JsonReader reader, Type objectType, StateParameters existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            JObject jo = JObject.Load(reader);
            StateParameters result = new StateParameters();

            result.InitialStateDistribution = jo["InitialStateDistribution"]?.ToObject<double[]>();
            result.LogTransitionProbabilities = jo["LogTransitionProbabilities"]?.ToObject<double[,]>();
            var observationsObj = (JObject)jo["Observations"];
            var observationsType = GetFromString(observationsObj["ObservationsType"]?.ToString());

            object[] kwargsArray = null;
            object[] paramsArray = [];

            if (observationsObj.ContainsKey("Kwargs"))
            {
                var kwargs = (JObject)observationsObj["Kwargs"];
                kwargsArray = kwargs.Properties()
                    .Select(p => p.Value.ToObject<object>())
                    .ToArray();
                if (kwargsArray.Count() == 0)
                {
                    kwargsArray = null;
                }
            }

            var observations = (ObservationsModel)Activator.CreateInstance(GetObservationsClassType(observationsType), kwargsArray);

            if (observationsObj.ContainsKey("Params"))
            {
                var paramsJArray = (JArray)observationsObj["Params"];
                var nParams = paramsJArray.Count;
                paramsArray = new object[nParams];
                for (int i = 0; i < nParams; i++)
                {
                    paramsArray[i] = NumpyHelper.NumpyParser.ParseString(paramsJArray[i].ToString(), typeof(double));
                }
            }

            observations.Params = paramsArray;
            result.Observations = observations;

            return result;
        }

        /// <inheritdoc/>
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
