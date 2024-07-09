using System;
using Python.Runtime;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public abstract class ObservationsBase
    {
        public abstract object[] Params { get; set; }
    }

    public abstract class ObservationsBase<T> : ObservationsBase where T : ObservationsBase<T>
    {
        public abstract override object[] Params { get; set; }
    }
}
