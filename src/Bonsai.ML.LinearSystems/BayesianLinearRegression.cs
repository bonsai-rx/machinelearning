using System;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reactive.Linq;
using System.Xml.Serialization;
using MathNet.Numerics.LinearAlgebra;

namespace Bonsai.ML.LinearSystems
{
    /// <summary>
    /// Represents an operator that performs bayesian linear regression on a sequence
    /// of temporal observations.
    /// </summary>
    [Combinator]
    [Description("Performs bayesian linear regression on a sequence of temporal observations.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class BayesianLinearRegression
    {
        /// <summary>
        /// Gets or sets the assumed fixed precision for the prior.
        /// </summary>
        [Description("The assumed fixed precision for the prior.")]
        public double PriorPrecision { get; set; }

        /// <summary>
        /// Gets or sets the assumed fixed precision for the likelihood.
        /// </summary>
        [Description("The assumed fixed precision for the likelihood.")]
        public double LikelihoodPrecision { get; set; }

        /// <summary>
        /// Gets or sets the initial mean estimate for the regression process.
        /// </summary>
        [TypeConverter(typeof(UnidimensionalArrayConverter))]
        [Description("The initial mean estimate for the regression process.")]
        public double[] M0 { get; set; }

        /// <summary>
        /// Gets or sets the initial variance estimate for the regression process.
        /// </summary>
        [XmlIgnore]
        [TypeConverter(typeof(MultidimensionalArrayConverter))]
        [Description("The initial mean estimate for the regression process.")]
        public double[,] S0 { get; set; }

        /// <summary>
        /// Gets or sets an XML representation of the image convolution kernel.
        /// </summary>
        [Browsable(false)]
        [XmlElement(nameof(S0))]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public string S0Xml
        {
            get { return ArrayConvert.ToString(S0, CultureInfo.InvariantCulture); }
            set { S0 = (double[,])ArrayConvert.ToArray(value, 2, typeof(double), CultureInfo.InvariantCulture); }
        }

        /// <summary>
        /// Updates the posterior given an observation and precision parameters.
        /// </summary>
        /// <param name="mn">The prior estimated mean.</param>
        /// <param name="Sn">The prior estimated variance.</param>
        /// <param name="phi">The new observation vector.</param>
        /// <param name="y">The temporal weighting for the observation vector.</param>
        /// <param name="alpha">The precision of the prior.</param>
        /// <param name="beta">The precision of the likelihood.</param>
        /// <returns>The estimated posterior.</returns>
        public static (Vector<double> mean, Matrix<double> cov) OnlineUpdate(Vector<double> mn, Matrix<double> Sn, Vector<double> phi, double y, double alpha, double beta)
        {
            Vector<double> aux1 = Sn.Multiply(phi);
            double aux2 = 1.0 / (1.0 / beta + phi.ToRowMatrix().Multiply(Sn).Multiply(phi)[0]);

            Matrix<double> Snp1 = Sn - aux2 * aux1.OuterProduct(aux1);
            Vector<double> mnp1 = beta * y * Snp1.Multiply(phi) + mn - aux2 * phi.DotProduct(mn) * Sn.Multiply(phi);

            return (mnp1, Snp1);
        }

        /// <summary>
        /// Performs bayesian linear regression on a sequence of temporal observations.
        /// </summary>
        /// <param name="source">The sequence of temporal observations.</param>
        /// <returns>
        /// The sequence of estimated posterior distributions, updated for
        /// each temporal observation.
        /// </returns>
        public IObservable<MultivariateGaussian> Process(IObservable<(double x, double t)> source)
        {
            return Observable.Defer(() =>
            {
                Vector<double> mn = Vector<double>.Build.DenseOfArray(M0);
                Matrix<double> Sn = Matrix<double>.Build.DenseOfArray(S0);
                double alpha = PriorPrecision;
                double beta = LikelihoodPrecision;
                return source.Select(observation =>
                {
                    double[] aux = new[] { 1, observation.x };
                    Vector<double> phi = Vector<double>.Build.DenseOfArray(aux);
                    (mn, Sn) = OnlineUpdate(mn, Sn, phi, observation.t, alpha, beta);
                    return new MultivariateGaussian { Mean = mn, Covariance = Sn };
                });
            });
        }
    }
}
