﻿#pragma checksum "..\..\..\MainWin.xaml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "C9F6409450EA07B98FE64C0420C5EFED073ED423"
//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.42000
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

using Classifier_Generator_GUI;
using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Controls.Ribbon;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Shell;


namespace Classifier_Generator_GUI {
    
    
    /// <summary>
    /// MainWin
    /// </summary>
    public partial class MainWin : System.Windows.Controls.Page, System.Windows.Markup.IComponentConnector {
        
        
        #line 52 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox train_set;
        
        #line default
        #line hidden
        
        
        #line 56 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox test_set;
        
        #line default
        #line hidden
        
        
        #line 60 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button select_data;
        
        #line default
        #line hidden
        
        
        #line 65 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button process_data;
        
        #line default
        #line hidden
        
        
        #line 70 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button optimized_summary_win;
        
        #line default
        #line hidden
        
        
        #line 75 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Button plain_summary_win;
        
        #line default
        #line hidden
        
        
        #line 80 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Label data_split_label;
        
        #line default
        #line hidden
        
        
        #line 84 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Slider data_split_size;
        
        #line default
        #line hidden
        
        
        #line 88 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox plain_select;
        
        #line default
        #line hidden
        
        
        #line 92 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox opt_select;
        
        #line default
        #line hidden
        
        
        #line 97 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Label data_path;
        
        #line default
        #line hidden
        
        
        #line 110 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox bin_classif_set;
        
        #line default
        #line hidden
        
        
        #line 114 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ItemsControl bItems;
        
        #line default
        #line hidden
        
        
        #line 115 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox rf;
        
        #line default
        #line hidden
        
        
        #line 116 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox hgbc;
        
        #line default
        #line hidden
        
        
        #line 117 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox ada;
        
        #line default
        #line hidden
        
        
        #line 118 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox qda;
        
        #line default
        #line hidden
        
        
        #line 119 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox lda;
        
        #line default
        #line hidden
        
        
        #line 120 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox mlp;
        
        #line default
        #line hidden
        
        
        #line 121 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox ridge;
        
        #line default
        #line hidden
        
        
        #line 122 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox pa;
        
        #line default
        #line hidden
        
        
        #line 123 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox sgd;
        
        #line default
        #line hidden
        
        
        #line 124 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox etc;
        
        #line default
        #line hidden
        
        
        #line 125 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox gnb;
        
        #line default
        #line hidden
        
        
        #line 126 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox mnb;
        
        #line default
        #line hidden
        
        
        #line 127 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox knn;
        
        #line default
        #line hidden
        
        
        #line 128 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox svc;
        
        #line default
        #line hidden
        
        
        #line 129 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox lsvc;
        
        #line default
        #line hidden
        
        
        #line 130 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox gp;
        
        #line default
        #line hidden
        
        
        #line 131 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox dt;
        
        #line default
        #line hidden
        
        
        #line 137 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox multi_classif_set;
        
        #line default
        #line hidden
        
        
        #line 141 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ItemsControl mItems;
        
        #line default
        #line hidden
        
        
        #line 142 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox rf2;
        
        #line default
        #line hidden
        
        
        #line 143 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox hgbc2;
        
        #line default
        #line hidden
        
        
        #line 144 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox qda2;
        
        #line default
        #line hidden
        
        
        #line 145 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox lda2;
        
        #line default
        #line hidden
        
        
        #line 146 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox mlp2;
        
        #line default
        #line hidden
        
        
        #line 147 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox ridge2;
        
        #line default
        #line hidden
        
        
        #line 148 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox gnb2;
        
        #line default
        #line hidden
        
        
        #line 149 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox knn2;
        
        #line default
        #line hidden
        
        
        #line 150 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox lsvc2;
        
        #line default
        #line hidden
        
        
        #line 151 "..\..\..\MainWin.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.CheckBox gp2;
        
        #line default
        #line hidden
        
        private bool _contentLoaded;
        
        /// <summary>
        /// InitializeComponent
        /// </summary>
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "8.0.1.0")]
        public void InitializeComponent() {
            if (_contentLoaded) {
                return;
            }
            _contentLoaded = true;
            System.Uri resourceLocater = new System.Uri("/Classifier Generator GUI;component/mainwin.xaml", System.UriKind.Relative);
            
            #line 1 "..\..\..\MainWin.xaml"
            System.Windows.Application.LoadComponent(this, resourceLocater);
            
            #line default
            #line hidden
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "8.0.1.0")]
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Design", "CA1033:InterfaceMethodsShouldBeCallableByChildTypes")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Maintainability", "CA1502:AvoidExcessiveComplexity")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1800:DoNotCastUnnecessarily")]
        void System.Windows.Markup.IComponentConnector.Connect(int connectionId, object target) {
            switch (connectionId)
            {
            case 1:
            this.train_set = ((System.Windows.Controls.ListBox)(target));
            return;
            case 2:
            this.test_set = ((System.Windows.Controls.ListBox)(target));
            return;
            case 3:
            this.select_data = ((System.Windows.Controls.Button)(target));
            
            #line 61 "..\..\..\MainWin.xaml"
            this.select_data.Click += new System.Windows.RoutedEventHandler(this.dataSelectButton);
            
            #line default
            #line hidden
            return;
            case 4:
            this.process_data = ((System.Windows.Controls.Button)(target));
            
            #line 66 "..\..\..\MainWin.xaml"
            this.process_data.Click += new System.Windows.RoutedEventHandler(this.dataProcessingButton);
            
            #line default
            #line hidden
            return;
            case 5:
            this.optimized_summary_win = ((System.Windows.Controls.Button)(target));
            
            #line 71 "..\..\..\MainWin.xaml"
            this.optimized_summary_win.Click += new System.Windows.RoutedEventHandler(this.optSummWin);
            
            #line default
            #line hidden
            return;
            case 6:
            this.plain_summary_win = ((System.Windows.Controls.Button)(target));
            
            #line 76 "..\..\..\MainWin.xaml"
            this.plain_summary_win.Click += new System.Windows.RoutedEventHandler(this.plainSummWin);
            
            #line default
            #line hidden
            return;
            case 7:
            this.data_split_label = ((System.Windows.Controls.Label)(target));
            return;
            case 8:
            this.data_split_size = ((System.Windows.Controls.Slider)(target));
            return;
            case 9:
            this.plain_select = ((System.Windows.Controls.CheckBox)(target));
            
            #line 88 "..\..\..\MainWin.xaml"
            this.plain_select.Checked += new System.Windows.RoutedEventHandler(this.PlainCheck);
            
            #line default
            #line hidden
            return;
            case 10:
            this.opt_select = ((System.Windows.Controls.CheckBox)(target));
            
            #line 92 "..\..\..\MainWin.xaml"
            this.opt_select.Checked += new System.Windows.RoutedEventHandler(this.OptCheck);
            
            #line default
            #line hidden
            return;
            case 11:
            this.data_path = ((System.Windows.Controls.Label)(target));
            return;
            case 12:
            this.bin_classif_set = ((System.Windows.Controls.ListBox)(target));
            return;
            case 13:
            this.bItems = ((System.Windows.Controls.ItemsControl)(target));
            return;
            case 14:
            this.rf = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 15:
            this.hgbc = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 16:
            this.ada = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 17:
            this.qda = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 18:
            this.lda = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 19:
            this.mlp = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 20:
            this.ridge = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 21:
            this.pa = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 22:
            this.sgd = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 23:
            this.etc = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 24:
            this.gnb = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 25:
            this.mnb = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 26:
            this.knn = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 27:
            this.svc = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 28:
            this.lsvc = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 29:
            this.gp = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 30:
            this.dt = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 31:
            this.multi_classif_set = ((System.Windows.Controls.ListBox)(target));
            return;
            case 32:
            this.mItems = ((System.Windows.Controls.ItemsControl)(target));
            return;
            case 33:
            this.rf2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 34:
            this.hgbc2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 35:
            this.qda2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 36:
            this.lda2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 37:
            this.mlp2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 38:
            this.ridge2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 39:
            this.gnb2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 40:
            this.knn2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 41:
            this.lsvc2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            case 42:
            this.gp2 = ((System.Windows.Controls.CheckBox)(target));
            return;
            }
            this._contentLoaded = true;
        }
    }
}

