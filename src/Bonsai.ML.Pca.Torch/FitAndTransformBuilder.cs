using System.ComponentModel;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that fits a PCA model and transforms the input data.
/// </summary>
[ResetCombinator]
[Combinator]
[Description("Fits a PCA model and transforms the input data.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FitAndTransformBuilder() : PcaModelBuilder<FitAndTransform>(new FitAndTransform()) { }
