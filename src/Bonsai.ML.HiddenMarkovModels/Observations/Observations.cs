using System;
using Python.Runtime;

namespace Bonsai.ML.HiddenMarkovModels.Observations
{
    public abstract class Observations
    {
        public abstract object[] Params { get; set; }
    }

    public abstract class Observations<T> : Observations where T : Observations<T>
    {
        public abstract override object[] Params { get; set; }
    }
}
