using System;
using System.ComponentModel;
using System.Reactive.Linq;
using static TorchSharp.torch;
using System.Xml.Serialization;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torchvision;

namespace Bonsai.ML.Torch.Vision
{
    [Combinator]
    [Description("")]
    [WorkflowElementCategory(ElementCategory.Source)]
    public class LoadPretrainedModel
    {
        public string Repository { get; set; } = "https://huggingface.co/";
        public Model Model { get; set; } = Model.AlexNet;
        public int? NumClasses { get; set; } = null;

        public IObservable<Module> Process()
        {
            var modelPath = Model.ToString() + ".pth";
            
            if (!System.IO.File.Exists(modelPath))
            {
                hub.download_url_to_file(
                    $"{Repository}yueyinqiu/vision-TorchSharp/resolve/main/{Model}_Weights.IMAGENET1K_V1",
                    modelPath);
            }

            Module model;

            switch (Model)
            {
                case Model.AlexNet:
                    if (NumClasses is null)
                    {
                        model = models.alexnet(weights_file: modelPath);
                    }
                    else
                    {
                        model = models.alexnet(
                            weights_file: modelPath, 
                            num_classes: NumClasses.Value,
                            skipfc: true
                        );
                    }
                    break;
                case Model.VGG16:
                    if (NumClasses is null)
                    {
                        model = models.vgg16(weights_file: modelPath);
                    }
                    else
                    {
                        model = models.vgg16(
                            weights_file: modelPath, 
                            num_classes: NumClasses.Value,
                            skipfc: true
                        );
                    }
                    break;
                case Model.VGG19:
                    if (NumClasses is null)
                    {
                        model = models.vgg19(weights_file: modelPath);
                    }
                    else
                    {
                        model = models.vgg19(
                            weights_file: modelPath, 
                            num_classes: NumClasses.Value,
                            skipfc: true
                        );
                    }
                    break;
                case Model.GoogLeNet:
                    if (NumClasses is null)
                    {
                        model = models.googlenet(weights_file: modelPath);
                    }
                    else
                    {
                        model = models.googlenet(
                            weights_file: modelPath, 
                            num_classes: NumClasses.Value,
                            skipfc: true
                        );
                    }
                    break;
                default:
                    throw new ArgumentException($"Model {Model} not supported.");
            }
            return Observable.Return(model);
        }
    }
}