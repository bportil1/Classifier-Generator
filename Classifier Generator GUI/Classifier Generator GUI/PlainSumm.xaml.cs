using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Xml.Linq;

namespace Classifier_Generator_GUI
{
    /// <summary>
    /// Interaction logic for PlainSumm.xaml
    /// </summary>
    public partial class PlainSumm : Page
    {
        static dynamic dataset_obj;
        static dynamic learning_obj;
        List<string> selected_algs;

        public PlainSumm(dynamic dataset_obj1, dynamic learning_obj1, List<string> selected_algs1)
        {
            dataset_obj = dataset_obj1;
            learning_obj = learning_obj1;
            selected_algs = selected_algs1;

            InitializeComponent();
            load_processed_algs(learning_obj, selected_algs);
            data_grid.DataContext = load_summary_data(dataset_obj);
            print_roc_curves(dataset_obj);
        }

        private void load_processed_algs(dynamic learning_obj, List<string> selected_algs)
        {
            for (int i = 0; i < selected_algs.Count; i++)
            {
                RadioButton cb = new RadioButton();
                cb.GroupName = "displayed_algs";
                cb.Name = selected_algs[i];
                cb.Content = get_alg_expanded_name(selected_algs[i]);
                cb.Checked += new RoutedEventHandler(load_images);
                cbItems.Items.Add(cb);
            }
        }

        private string get_alg_expanded_name(string abbr)
        {
            Dictionary<string, string> alg_names = new Dictionary<string, string>()
            {
                { "knn","K-Nearest Neighbors"},
                {"lsvc", "Linear SVC"},
                {"svc", "SVC"},
                { "gp", "Gaussian Processes"},
                { "dt", "Decision Tree"},
                { "rf", "Random Forest"},
                { "ada", "AdaBoost"},
                { "gnb", "Gaussian Naive Bayes"},
                { "mnb", "Multinomial Naive Bayes"},
                { "compnb", "Complement Naive Bayes"},
                { "bnb", "Bernoulli Naive Bayes"},
                { "qda", "Quadratic Discriminant Analysis"},
                { "lda", "Linear Discriminant Analysis"},
                { "mlp", "Multilayer Perceptron"},
                { "ridge", "Ridge"},
                { "pa", "Passive Aggressive"},
                { "sgd", "SGD"},
                { "hgbc", "Histogram Gradient Boosting Classifier"},
                { "etc", "Extra Trees Classifier"},
                { "if", "Isolation Forest"}
            };
            return alg_names[abbr];
        }

        private void print_roc_curves(dynamic dataset_obj)
        {
            if (dataset_obj.classif_type == "binary")
            {
                string file = dataset_obj.data_path + "/supervised_methods_evaluation_results/roc_data_plain_summary.png";
                group_roc_curves_img.Source = new BitmapImage(new Uri(@file));
            }
        }

        private DataTable load_summary_data(dynamic dataset_obj)
        {
            string data_file = dataset_obj.data_path + "/supervised_methods_evaluation_results/plain_summary.csv";

            StreamReader sr = new StreamReader(data_file);
            DataTable td = new DataTable();
            string line = "";
            int idx1 = 0;
            int row_len;
            while (line != null)
            {
                line = sr.ReadLine();
                if (line != null)
                {
                    string[] indv = line.Split(",");
                    row_len = indv.Length;
                    if (idx1 == 0)
                    {
                        foreach (var ind in indv)
                        {
                            td.Columns.Add(ind, typeof(string));
                        }
                        idx1++;
                    }

                    else
                    {
                        DataRow newRow = td.NewRow();
                        for (int i = 0; i < row_len; i++)
                        {
                            newRow[i] = indv[i];
                        }
                        td.Rows.Add(newRow);
                    }
                }
            }
            return td;
        }

        void load_images(object sender, RoutedEventArgs e)
        {
            RadioButton rb = (RadioButton)sender;

            //System.Diagnostics.Debug.Print("prr:  " + rb.Name);

            if (dataset_obj.classif_type == "binary")
            {
                load_indv_roc_curve(rb, dataset_obj);
            }

            load_confusion_matrix(rb, dataset_obj);
            load_decision_boundary(rb, dataset_obj);

        }

        private void load_decision_boundary(RadioButton rb,
                                            dynamic dataset_obj)
        {
            string rbname = rb.Name;
            if (rb != null)
            {
                if (rb.IsChecked == true)
                {
                    string datapath = dataset_obj.data_path
                                      + "/supervised_methods_evaluation_results/"
                                       + rbname + "/plain_results/" + rbname
                                      + " dbd.png";

                    decision_boundary_img.Source = new BitmapImage(new Uri(datapath));
                }
            }
        }

        private void load_confusion_matrix(RadioButton rb,
                                    dynamic dataset_obj)
        {
            string rbname = rb.Name;
            if (rb != null)
            {
                if (rb.IsChecked == true)
                {
                    string datapath = dataset_obj.data_path
                                      + "/supervised_methods_evaluation_results/"
                                      + rbname + "/plain_results/" + rbname
                                      + " cm.png";

                    confusion_matrix_img.Source = new BitmapImage(new Uri(datapath));
                }
            }
        }

        private void load_indv_roc_curve(RadioButton rb,
                                         dynamic dataset_obj)
        {
            string rbname = rb.Name;

            if (rb != null)
            {
                if (rb.IsChecked == true)
                {
                    string datapath = dataset_obj.data_path
                                      + "/supervised_methods_evaluation_results/"
                                      + rbname + "/plain_results/" + rbname
                                      + "_ras.png";

                    indv_roc_curve_img.Source = new BitmapImage(new Uri(datapath));
                }
            }
        }
    }
}
