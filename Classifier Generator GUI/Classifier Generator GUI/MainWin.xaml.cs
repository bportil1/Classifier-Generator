using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.Eventing.Reader;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
using static Classifier_Generator_GUI.PythonInterop;

namespace Classifier_Generator_GUI
{
    /// <summary>
    /// Interaction logic for MainWin.xaml
    /// </summary>
    public partial class MainWin : Page
    {

        static dynamic dataset_obj;
        static dynamic learning_obj;
        static dynamic ll;
        static bool opt = false;
        static bool plain = false;
        static bool loaded = false;
        static bool opt_processed = false;
        static bool plain_processed = false;
        static List<string> selected_algs;

        public MainWin(Frame mainFrame)
        {
            InitializeComponent();
            MainWindow mainWindow = (MainWindow)Window.GetWindow(this);
            Initialize();
            //PythonEngine.BeginAllowThreads();
            using (Py.GIL())
            {
                ll = Py.Import("learningScripts.learning_lib");
                dataset_obj = ll.dataset();
            }

        }
        
        void dataSelectButton(object sender, RoutedEventArgs e)
        {
            if (loaded)
            {
                dataset_obj = ll.dataset();
                select_data.Background = Brushes.Gray;
                process_data.Background = Brushes.Gray;
            }

            using (Py.GIL())
            {
                dataset_obj.select_data();
                data_path.Content = "Current Data: " + dataset_obj.data_path;
                dataset_obj.load_data();
                select_data.Background = Brushes.Red;
                dataset_obj.split_data(data_split_size.Value/100);

                string complete_data = ((object)dataset_obj.complete_data).ToString();
                DataTable data_col = collect_selected_data(complete_data);
                data_grid.DataContext = data_col;

                dataset_obj.downsize_data();
                loaded = true;
                opt_processed = false;
                plain_processed = false;
            }
            select_data.Background = Brushes.Red;
        }

        void dataProcessingButton(object sender, RoutedEventArgs e)
        { 
            if (loaded == true)
            {
                if (dataset_obj.classif_type == "binary")
                {
                    binAlgCol();
                }
                else if (dataset_obj.classif_type == "multiclass")
                {
                    multiAlgCol();
                }
                using (Py.GIL())
                {
                    if (opt && selected_algs.Any())
                    {
                        learning_obj = ll.learning(dataset_obj, "optimized", selected_algs);
                        learning_obj.supervised_learning();
                        opt_processed = true;
                    }
                    else if (opt)
                    {
                        learning_obj = ll.learning(dataset_obj, "optimized");
                        learning_obj.supervised_learning();
                        opt_processed = true;
                    }
                    if (plain && selected_algs.Any())
                    {
                        learning_obj = ll.learning(dataset_obj, "plain", selected_algs);
                        learning_obj.supervised_learning();
                        plain_processed = true;
                    }
                    else if (plain)
                    {
                        learning_obj = ll.learning(dataset_obj, "plain");
                        learning_obj.supervised_learning();
                        plain_processed = true;
                    }
                    /*
                    if (!selected_algs.Any())
                    {
                        learning_obj = ll.learning(dataset_obj, "plain");
                        learning_obj.supervised_learning();
                    }
                    */
                }
            }
            else
            {
                MessageBox.Show("Select a dataset to process first");
            }
            process_data.Background = Brushes.Red;
            
        }

        void optSummWin(object sender, RoutedEventArgs e)
        {
            MainWindow mainWindow = (MainWindow)Window.GetWindow(this);
            if (opt_processed)
            {
                mainWindow.mainFrame.Navigate(new OptSumm(dataset_obj, learning_obj, selected_algs));
            }
            else
            {
                MessageBox.Show("Process data first");
            }
        }

        void plainSummWin(object sender, RoutedEventArgs e)
        {
            MainWindow mainWindow = (MainWindow)Window.GetWindow(this);
            if (plain_processed)
            {
                mainWindow.mainFrame.Navigate(new PlainSumm(dataset_obj, learning_obj, selected_algs));
            }
            else
            {
                MessageBox.Show("Process data first");
            }

        }

        private void PlainCheck(object sender, RoutedEventArgs e)
        {
            plain = true;
        }

        private void OptCheck(object sender, RoutedEventArgs e)
        {
            opt = true;
        }
        private void PlainUncheck(object sender, RoutedEventArgs e)
        {
            plain = false;
        }

        private void OptUncheck(object sender, RoutedEventArgs e)
        {
            opt = false; 
        }

        private void binAlgCol()
        {
            selected_algs = new List<string>();
            foreach (CheckBox cbox in bItems.Items)
            {
                if (cbox.IsChecked == true)
                {
                    selected_algs.Add(cbox.Name);
                }
            }

        }
        private void multiAlgCol()
        {
            selected_algs = new List<string>();
            foreach (CheckBox cbox in mItems.Items)
            {
                if (cbox.IsChecked == true)
                {
                    selected_algs.Add(cbox.Name.Remove(cbox.Name.Length - 1, 1));
                }
            }

        }
        
        private DataTable collect_selected_data(string data){

            DataTable td = new DataTable();

            int idx1 = 0;
            int row_len;
            using (System.IO.StringReader reader = new System.IO.StringReader(data))
            {
                string line = "";
                while ( line  != null)
                {
                    line = reader.ReadLine();
                    if (line != null && line != ""  && !(line.Contains("[")))
                    { 
                        string[] indv = line.Split(null as char[], StringSplitOptions.RemoveEmptyEntries);
                        row_len = indv.Length;
                        if (idx1 == 0)
                        {
                            td.Columns.Add("Index");
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
            }
            return td;
        }

    }
}
