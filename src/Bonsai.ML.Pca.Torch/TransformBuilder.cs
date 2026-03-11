using System.ComponentModel;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that transforms the input data using a PCA model.
/// </summary>
[ResetCombinator]
[Combinator]
[Description("Transforms the input data using a PCA model.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class TransformBuilder() : PcaModelBuilder<Transform>(new Transform()) { }
