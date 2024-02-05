using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Policy;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;
using System.Xml.Serialization;
using static System.Formats.Asn1.AsnWriter;

namespace Classifier_Generator_GUI
{
    public class PythonInterop
    {

        public static void Initialize()
        {
            string pythonDll = @"c:Users/Karkaras/AppData/Local/Programs/Python/Python38/python38.dll";
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", pythonDll);
            PythonEngine.Initialize();
            //PythonEngine.BeginAllowThreads();
        }


    }
}
