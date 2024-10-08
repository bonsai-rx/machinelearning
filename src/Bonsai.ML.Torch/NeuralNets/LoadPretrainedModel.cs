using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;
using static TorchSharp.torch.nn;
using Bonsai.Expressions;

namespace Bonsai.ML.Torch.NeuralNets
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LoadPretrainedModel
    {
        public Models.PretrainedModels ModelName { get; set; }
        [XmlIgnore]
        public Device Device { get; set; }

        [Editor("Bonsai.Design.OpenFileNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
        public string? ModelWeightsPath { get; set; }

        private int numClasses = 10;

        public IObservable<Module> Process()
        {
            Module<Tensor,Tensor> model = null;
            var modelName = ModelName.ToString().ToLower();
            var device = Device;

            switch (modelName)
            {
                case "alexnet":
                    model = new Models.AlexNet(modelName, numClasses, device);
                    if (ModelWeightsPath is not null) model.load(ModelWeightsPath);
                    break;
                case "mobilenet":
                    model = new Models.MobileNet(modelName, numClasses, device);
                    if (ModelWeightsPath is not null) model.load(ModelWeightsPath);
                    break;
                case "mnist":
                    model = new Models.MNIST(modelName, device);
                    if (ModelWeightsPath is not null) model.load(ModelWeightsPath);
                    break;
                default:
                    throw new ArgumentException($"Model {modelName} not supported.");
            }
            Console.WriteLine($"Model: {model}");

            return Observable.Defer(() => {
                return Observable.Return(model);
            });
        }
    }
}