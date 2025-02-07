using System;
using System.ComponentModel;
using System.Reactive.Linq;
using PointProcessDecoder.Core;
using static TorchSharp.torch;

namespace Bonsai.ML.PointProcessDecoder;

/// <summary>
/// Saves the state of the point process model.
/// </summary>
[Combinator]
[WorkflowElementCategory(ElementCategory.Sink)]
[Description("Saves the state of the point process model.")]
public class SavePointProcessModel
{
    /// <summary>
    /// The path to the folder where the state of the point process model will be saved.
    /// </summary>
    [Editor("Bonsai.Design.FolderNameEditor, Bonsai.Design", DesignTypes.UITypeEditor)]
    [Description("The path to the folder where the state of the point process model will be saved.")]
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// If true, the contents of the folder will be overwritten if it already exists.
    /// </summary>
    [Description("If true, the contents of the folder will be overwritten if it already exists.")]
    public bool Overwrite { get; set; } = false;

    /// <summary>
    /// Specifies the type of suffix to add to the save path.
    /// </summary>
    [Description("Specifies the type of suffix to add to the save path.")]
    public SuffixType AddSuffix { get; set; } = SuffixType.None;

    /// <summary>
    /// The name of the point process model to save.
    /// </summary>
    [TypeConverter(typeof(PointProcessModelNameConverter))]
    [Description("The name of the point process model to save.")]
    public string Model { get; set; } = string.Empty;

    /// <summary>
    /// Saves the state of the point process model.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="source"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public IObservable<T> Process<T>(IObservable<T> source)
    {
        return source.Do(_ => {

            var path = AddSuffix switch
            {
                SuffixType.DateTime => System.IO.Path.Combine(Path, $"{DateTime.Now:yyyyMMddHHmmss}"),
                SuffixType.Guid => System.IO.Path.Combine(Path, Guid.NewGuid().ToString()),
                _ => Path
            };

            if (string.IsNullOrEmpty(path))
            {
                throw new InvalidOperationException("The save path is not specified.");
            }

            if (System.IO.Directory.Exists(path))
            {
                if (Overwrite)
                {
                    System.IO.Directory.Delete(path, true);
                }
                else
                {
                    throw new InvalidOperationException("The save path already exists and overwrite is set to False.");
                }
            }

            var model = PointProcessModelManager.GetModel(Model);

            model.Save(path);
        });
    }

    /// <summary>
    /// Specifies the type of suffix to add to the save path.
    /// </summary>
    public enum SuffixType
    {
        /// <summary>
        /// No suffix is added to the save path.
        /// </summary>
        None,

        /// <summary>
        /// A suffix with the current date and time is added to the save path.
        /// </summary>
        DateTime,

        /// <summary>
        /// A suffix with a unique identifier is added to the save path.
        /// </summary>
        Guid
    }
}