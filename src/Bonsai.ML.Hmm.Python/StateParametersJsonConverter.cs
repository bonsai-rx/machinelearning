using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Linq;
using Bonsai.ML.Hmm.Python.Observations;
using Bonsai.ML.Hmm.Python.Transitions;
using Bonsai.ML.Data;

namespace Bonsai.ML.Hmm.Python
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
            StateParameters result = new StateParameters
            {
                InitialStateDistribution = jo["InitialStateDistribution"]?.ToObject<double[]>()
            };

            var transitionsObj = (JObject)jo["Transitions"];
            var transitionModelType = TransitionModelLookup.GetFromString(transitionsObj["TransitionModelType"]?.ToString());

            object[] transitionsKwargsArray = null;
            object[] transitionsParamsArray = [];

            if (transitionsObj.ContainsKey("Kwargs"))
            {
                var kwargs = (JObject)transitionsObj["Kwargs"];
                transitionsKwargsArray = kwargs.Properties()
                    .Select(p => p.Value.ToObject<object>())
                    .ToArray();
                if (transitionsKwargsArray.Count() == 0)
                {
                    transitionsKwargsArray = null;
                }
            }

            var transitions = (TransitionModel)Activator.CreateInstance(TransitionModelLookup.GetTransitionsClassType(transitionModelType), transitionsKwargsArray);

            if (transitionsObj.ContainsKey("Params"))
            {
                var paramsJArray = (JArray)transitionsObj["Params"];
                var nParams = paramsJArray.Count;
                transitionsParamsArray = new object[nParams];
                for (int i = 0; i < nParams; i++)
                {
                    transitionsParamsArray[i] = JsonDataHelper.ParseToken(paramsJArray[i], typeof(double));
                }
            }

            transitions.Params = transitionsParamsArray;
            result.Transitions = transitions;
            
            var observationsObj = (JObject)jo["Observations"];
            var observationModelType = ObservationModelLookup.GetFromString(observationsObj["ObservationModelType"]?.ToString());

            object[] observationsKwargsArray = null;
            object[] observationsParamsArray = [];

            if (observationsObj.ContainsKey("Kwargs"))
            {
                var kwargs = (JObject)observationsObj["Kwargs"];
                observationsKwargsArray = kwargs.Properties()
                    .Select(p => p.Value.ToObject<object>())
                    .ToArray();
                if (observationsKwargsArray.Count() == 0)
                {
                    observationsKwargsArray = null;
                }
            }

            var observations = (ObservationModel)Activator.CreateInstance(ObservationModelLookup.GetObservationsClassType(observationModelType), observationsKwargsArray);

            if (observationsObj.ContainsKey("Params"))
            {
                var paramsJArray = (JArray)observationsObj["Params"];
                var nParams = paramsJArray.Count;
                observationsParamsArray = new object[nParams];
                for (int i = 0; i < nParams; i++)
                {
                    observationsParamsArray[i] = JsonDataHelper.ParseToken(paramsJArray[i], typeof(double));
                }
            }

            observations.Params = observationsParamsArray;
            result.Observations = observations;

            return result;
        }

        /// <inheritdoc/>
        public override void WriteJson(JsonWriter writer, StateParameters value, JsonSerializer serializer)
        {
            writer.WriteStartObject();

            writer.WritePropertyName("InitialStateDistribution");
            serializer.Serialize(writer, value.InitialStateDistribution);

            writer.WritePropertyName("Transitions");
            serializer.Serialize(writer, value.Transitions);

            writer.WritePropertyName("Observations");
            serializer.Serialize(writer, value.Observations);

            writer.WriteEndObject();
        }
    }
}
