   M S B A M L   `   `   `  ����S  OClassifier Generator GUI, Version=1.0.0.0, Culture=neutral, PublicKeyToken=nullC&clr-namespace:Classifier_Generator_GUIClassifier_Generator_GUI  5      (    "Classifier_Generator_GUI.PlainSumm   5      W SPresentationCore, Version=7.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=7.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=7.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=7.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35F 9http://schemas.microsoft.com/winfx/2006/xaml/presentation     5      R NSystem.Xaml, Version=7.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5      Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5      @d2http://schemas.microsoft.com/expression/blend/2008     5      8local&clr-namespace:Classifier_Generator_GUI     5        N� Title$   Default Attributes Summary��5	      .��5      �  � RowDefinitions 5   
   �� 5      $
��Auto �6   �� 5      $
��Auto �6   �� 5      $
��Auto �6    � ColumnDefinitions 5   
   �� 5      $
��Auto �6   �� 5      $
��Auto �6   �� 5      $
��Auto �6   .+�5   
   �� $	��100(�6   $��	-6,10,0,0q�6    J� RenderTransformOrigin$ 0.5,0.5&�62   
����  �
�������$��Italic$�5   9   $
��Left=�5      $
��1392��6+   $��3��5      $��0��6$   $	��143��5      $	��Top=�6   .��65   Plain Attributes Summary5   '   �� 5!   
   $	��140��6   
����  �$��-6,794,0,-130q�61   $��3��6H   $��3��6\   �� 5#   
   -   5        $��learning_methods��5#      $��10,208,0,555q�6*    �� SelectionMode$ Single=�5$      $
��Left=�6(   $	��2in��5%      $��4.5in��6!   $	��400��62   .@�5&      � .+�5'      �� $	��2in��6   $��Visible=�6/   .��5(      �� -   5        $��cbItems��5(   #   5*      5+      5,      5.      �� 5/   
   -   5        $��confusion_matrix_img��5/   9   $%��image-not-available-768x448.png��6   $	��320��50      $
��Fill=�6    $��2��51      $��
25,158,0,0q�6#   $	��Top=�67   $
��Left=�6O   $	��492��6j   �� 54   
   -   5        $��decision_boundary_img��54   9   $%��image-not-available-768x448.png��6   $	��320��55      $	��492��6    $
��Fill=�6,   $��2��56      $��25,483,620,0q�6#   $	��Top=�69   �� 59   
   -   5        $��indv_roc_curve_img��59   9   $%��image-not-available-768x448.png��6   $	��320��5:      $	��400��6    $
��Fill=�6,   $��2��5;      $��558,484,179,0q�6#   $	��Top=�6:   �� 5>   
   -   5        $��group_roc_curves_img��5>   9   $%��image-not-available-768x448.png��6   $	��320��5?      $	��400��6    $
��Fill=�6,   $��558,159,179,0q�6;   $	��Top=�6R   $��2��6j   &   System.Windows.Controls.DataGrid  5B   
   -   5          Name$ 	data_grid��5B      $��7.0in��5C      $	��7in��6"   $   HorizontalScrollBarVisibility$ Visible=�5D      "   VerticalScrollBarVisibility$ Visible=�5E         AlternatingRowBackground
 ��·�	 �� AlternationCount$	 2��5G      $��
0,474,10,0q�6&   $��Right=�5H      $	��384��5I      
   
IsReadOnly$

 True��6   $	��320��6/   $	��Top=�6<   ��5B   &   �� 5I   <   5K      5L      