using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Linq;
using Bonsai.ML.HiddenMarkovModels.Observations;
using Bonsai.ML.HiddenMarkovModels.Transitions;
using Bonsai.ML.Python;

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
            StateParameters result = new StateParameters
            {
                InitialStateDistribution = jo["InitialStateDistribution"]?.ToObject<double[]>()
            };

            var transitionsObj = (JObject)jo["Transitions"];
            var transitionsModelType = TransitionsModelLookup.GetFromString(transitionsObj["TransitionsModelType"]?.ToString());

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

            var transitions = (TransitionsModel)Activator.CreateInstance(TransitionsModelLookup.GetTransitionsClassType(transitionsModelType), transitionsKwargsArray);

            if (transitionsObj.ContainsKey("Params"))
            {
                var paramsJArray = (JArray)transitionsObj["Params"];
                var nParams = paramsJArray.Count;
                transitionsParamsArray = new object[nParams];
                for (int i = 0; i < nParams; i++)
                {
                    transitionsParamsArray[i] = NumpyHelper.NumpyParser.ParseString(paramsJArray[i].ToString(), typeof(double));
                }
            }

            transitions.Params = transitionsParamsArray;
            result.Transitions = transitions;
            
            var observationsObj = (JObject)jo["Observations"];
            var observationsModelType = ObservationsModelLookup.GetFromString(observationsObj["ObservationsModelType"]?.ToString());

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

            var observations = (ObservationsModel)Activator.CreateInstance(ObservationsModelLookup.GetObservationsClassType(observationsModelType), observationsKwargsArray);

            if (observationsObj.ContainsKey("Params"))
            {
                var paramsJArray = (JArray)observationsObj["Params"];
                var nParams = paramsJArray.Count;
                observationsParamsArray = new object[nParams];
                for (int i = 0; i < nParams; i++)
                {
                    observationsParamsArray[i] = NumpyHelper.NumpyParser.ParseString(paramsJArray[i].ToString(), typeof(double));
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
