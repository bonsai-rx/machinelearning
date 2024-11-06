using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;

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
        public string ModelWeightsPath { get; set; }

        private int numClasses = 10;

        public IObservable<ITorchModule> Process()
        {
            nn.Module<Tensor,Tensor> module = null;
            var modelName = ModelName.ToString().ToLower();
            var device = Device;

            switch (modelName)
            {
                case "alexnet":
                    module = new Models.AlexNet(modelName, numClasses, device);
                    if (ModelWeightsPath is not null) module.load(ModelWeightsPath);
                    break;
                case "mobilenet":
                    module = new Models.MobileNet(modelName, numClasses, device);
                    if (ModelWeightsPath is not null) module.load(ModelWeightsPath);
                    break;
                case "mnist":
                    module = new Models.MNIST(modelName, device);
                    if (ModelWeightsPath is not null) module.load(ModelWeightsPath);
                    break;
                default:
                    throw new ArgumentException($"Model {modelName} not supported.");
            }

            var torchModule = new TorchModuleAdapter(module);
            return Observable.Defer(() => {
                return Observable.Return((ITorchModule)torchModule);
            });
        }
    }
}