
namespace Bonsai.ML
{
    /// <summary>
    /// Bonsai.ML model reference base class
    /// </summary>
    public class ModelReference
    {
        /// <summary>
        /// Gets or sets the name of the referenced model.
        /// </summary>
        public string Name { get ; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelReference"/> class
        /// with the specified name.
        /// </summary>
        /// <param name="name">The name of the referenced model.</param>
        public ModelReference(string name)
        {
            Name = name;
        }
    }
}
