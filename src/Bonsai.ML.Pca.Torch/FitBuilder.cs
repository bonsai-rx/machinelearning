using System.ComponentModel;

namespace Bonsai.ML.Pca.Torch;

/// <summary>
/// Represents an operator that fits a PCA model to the input data.
/// </summary>
[ResetCombinator]
[Combinator]
[Description("Fits a PCA model to the input data.")]
[WorkflowElementCategory(ElementCategory.Sink)]
public class FitBuilder() : PcaModelBuilder<Fit>(new Fit()) { }
