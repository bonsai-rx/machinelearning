using System.ComponentModel;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that reconstructs the input data using a PCA model.
/// </summary>
[ResetCombinator]
[Combinator]
[Description("Reconstructs the input data using a PCA model.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ReconstructBuilder() : PcaModelBuilder<Reconstruct>(new Reconstruct()) { }
