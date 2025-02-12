using System;
using System.Windows.Forms;

namespace Bonsai.ML.PointProcessDecoder.Design;

internal class ToolStripNumericUpDown : ToolStripControlHost
{
    public ToolStripNumericUpDown()
        : base(new NumericUpDown())
    {
    }

    public NumericUpDown NumericUpDown
    {
        get { return Control as NumericUpDown; }
    }

    public int DecimalPlaces
    {
        get { return NumericUpDown.DecimalPlaces; }
        set { NumericUpDown.DecimalPlaces = value; }
    }

    public decimal Value
    {
        get { return NumericUpDown.Value; }
        set { NumericUpDown.Value = value; }
    }

    public decimal Minimum
    {
        get { return NumericUpDown.Minimum; }
        set { NumericUpDown.Minimum = value; }
    }

    public decimal Maximum
    {
        get { return NumericUpDown.Maximum; }
        set { NumericUpDown.Maximum = value; }
    }

    public event EventHandler ValueChanged
    {
        add { NumericUpDown.ValueChanged += value; }
        remove { NumericUpDown.ValueChanged -= value; }
    }
}
