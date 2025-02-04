using System;
using System.ComponentModel;
using System.Xml.Serialization;
using System.Linq;
using System.Reactive.Linq;

using static TorchSharp.torch;

using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Core.Likelihood;
using System.IO;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Loads a point process model from a saved state.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Source)]
[Description("Loads a point process model from a saved state.")]
public class LoadPointProcessModel
{
    private string name = "PointProcessModel";

    /// <summary>
    /// Gets or sets the name of the point process model.
    /// </summary>
    [Description("The name of the point process model.")]
    public string Name
    {
        get
        {
            return name;
        }
        set
        {
            name = value;
        }
    }

    Device? device = null;
    /// <summary>
    /// Gets or sets the device used to run the neural decoding model.
    /// </summary>
    [XmlIgnore]
    [Description("The device used to run the neural decoding model.")]
    public Device? Device
    {
        get
        {
            return device;
        }
        set
        {
            device = value;
        }
    }

    /// <summary>
    /// The path to the folder where the state of the point process model was saved.
    /// </summary>
    [Editor("Bonsai.Design.FolderNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    [Description("The path to the folder where the state of the point process model was saved.")]
    public string Path { get; set; } = string.Empty;


    /// <summary>
    /// Creates a new neural decoding model based on point processes using Bayesian state space models.
    /// </summary>
    /// <returns></returns>
    public IObservable<PointProcessModel> Process()
    {
        return Observable.Using(
            () => {

                if (string.IsNullOrEmpty(Path))
                {
                    throw new InvalidOperationException("The save path is not specified.");
                }

                if (!Directory.Exists(Path))
                {
                    throw new InvalidOperationException("The save path does not exist.");
                }

                return PointProcessModelManager.Load(
                    name: name,
                    path: Path,
                    device: device
                );    
            }, resource => Observable.Return(resource.Model)
                .Concat(Observable.Never(resource.Model))
                .Finally(resource.Dispose));
    }
}