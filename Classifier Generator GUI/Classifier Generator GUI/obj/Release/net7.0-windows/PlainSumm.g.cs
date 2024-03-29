﻿#pragma checksum "..\..\..\PlainSumm.xaml" "{ff1816ec-aa5e-4d10-87f7-6f4963833460}" "435A3D814388E9AD00210A934B86BC24D7CAC495"
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
    /// PlainSumm
    /// </summary>
    public partial class PlainSumm : System.Windows.Controls.Page, System.Windows.Markup.IComponentConnector {
        
        
        #line 15 "..\..\..\PlainSumm.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox Summary;
        
        #line default
        #line hidden
        
        
        #line 19 "..\..\..\PlainSumm.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox metrics_list;
        
        #line default
        #line hidden
        
        
        #line 23 "..\..\..\PlainSumm.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox acc_res;
        
        #line default
        #line hidden
        
        
        #line 27 "..\..\..\PlainSumm.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox visual_rep;
        
        #line default
        #line hidden
        
        
        #line 31 "..\..\..\PlainSumm.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox roc_curves;
        
        #line default
        #line hidden
        
        
        #line 35 "..\..\..\PlainSumm.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.ListBox decision_boundary_by_alg;
        
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
            System.Uri resourceLocater = new System.Uri("/Classifier Generator GUI;component/plainsumm.xaml", System.UriKind.Relative);
            
            #line 1 "..\..\..\PlainSumm.xaml"
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
            this.Summary = ((System.Windows.Controls.ListBox)(target));
            
            #line 17 "..\..\..\PlainSumm.xaml"
            this.Summary.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.Summary_SelectionChanged);
            
            #line default
            #line hidden
            return;
            case 2:
            this.metrics_list = ((System.Windows.Controls.ListBox)(target));
            
            #line 21 "..\..\..\PlainSumm.xaml"
            this.metrics_list.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.metrics_list_SelectionChanged);
            
            #line default
            #line hidden
            return;
            case 3:
            this.acc_res = ((System.Windows.Controls.ListBox)(target));
            
            #line 25 "..\..\..\PlainSumm.xaml"
            this.acc_res.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.Summary_SelectionChanged);
            
            #line default
            #line hidden
            return;
            case 4:
            this.visual_rep = ((System.Windows.Controls.ListBox)(target));
            
            #line 29 "..\..\..\PlainSumm.xaml"
            this.visual_rep.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.metrics_list_SelectionChanged);
            
            #line default
            #line hidden
            return;
            case 5:
            this.roc_curves = ((System.Windows.Controls.ListBox)(target));
            
            #line 33 "..\..\..\PlainSumm.xaml"
            this.roc_curves.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.Summary_SelectionChanged);
            
            #line default
            #line hidden
            return;
            case 6:
            this.decision_boundary_by_alg = ((System.Windows.Controls.ListBox)(target));
            
            #line 37 "..\..\..\PlainSumm.xaml"
            this.decision_boundary_by_alg.SelectionChanged += new System.Windows.Controls.SelectionChangedEventHandler(this.metrics_list_SelectionChanged);
            
            #line default
            #line hidden
            return;
            }
            this._contentLoaded = true;
        }
    }
}

