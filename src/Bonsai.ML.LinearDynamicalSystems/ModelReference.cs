using System.ComponentModel;

namespace Bonsai.ML.LinearDynamicalSystems
{

    /// <summary>
    /// Bonsai LDS model reference base class
    /// </summary>
    public class ModelReference
    {
        /// <summary>
        /// The name of the model
        /// </summary>
        [Description("Name of the model")]
        public string Name { get ; set; }

        public ModelReference(string name)
        {
            Name = name;
        }
    }
}
