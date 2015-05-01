using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace FeaturesReduction
{
    public partial class Main : Form
    {
        public Main()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            List<List<double>> l = new List<List<double>>();
            List<double> inpi = new List<double>();
            Random rnd = new Random();

            for (int i = 0; i < 2500; i++)
                inpi.Add(rnd.NextDouble());
            
            for (int i = 0; i < 50; i++) {
                l.Add(new List<double>());

                for (int j = 0; j < 2500; j++) {
                    l[i].Add(rnd.NextDouble());
                }
            }
            GeneralizedHebbianLearningPCA ghlp = new GeneralizedHebbianLearningPCA(inpi, 256, 0.001);
            ghlp.train(100, l);

            
        }
    }
}
