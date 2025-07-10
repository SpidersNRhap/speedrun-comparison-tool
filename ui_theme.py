import tkinter as tk
from tkinter import ttk

class UITheme:
    def __init__(self):
        self.dark_bg = "#0f0f0f"
        self.dark_fg = "#e0e0e0"
        self.dark_select_bg = "#2a2a2a"
        self.dark_entry_bg = "#1a1a1a"
        self.dark_button_bg = "#2a2a2a"
        self.dark_frame_bg = "#0f0f0f"
        self.dark_text_bg = "#1a1a1a"

    def setup_dark_theme(self, root):
        """Configure dark theme for the application"""
        style = ttk.Style()

        style.theme_use('clam')

        style.configure('TFrame', background=self.dark_bg)

        try:
            style.element_create('Dark.Labelframe.border', 'from', 'default')
            style.layout('Dark.TLabelframe', [
                ('Dark.Labelframe.border', {'sticky': 'nswe', 'children': [
                    ('Labelframe.padding', {'sticky': 'nswe', 'children': [
                        ('Labelframe.label', {'side': 'left', 'sticky': ''}),
                    ]})
                ]})
            ])

            style.configure('Dark.TLabelframe', 
                           background=self.dark_bg, 
                           foreground=self.dark_fg,
                           bordercolor=self.dark_bg,
                           darkcolor=self.dark_bg,
                           lightcolor=self.dark_bg,
                           focuscolor=self.dark_bg,
                           borderwidth=0,
                           relief='flat')

            style.map('Dark.TLabelframe',
                     background=[('', self.dark_bg)],
                     bordercolor=[('', self.dark_bg)],
                     lightcolor=[('', self.dark_bg)],
                     darkcolor=[('', self.dark_bg)],
                     focuscolor=[('', self.dark_bg)])

        except Exception as e:
            print(f"Advanced labelframe styling failed: {e}")
            style.configure('Dark.TLabelframe', 
                           background=self.dark_bg, 
                           foreground=self.dark_fg,
                           borderwidth=0,
                           relief='flat')

        style.configure('Dark.TLabelframe.Label', 
                       background=self.dark_bg, 
                       foreground=self.dark_fg,
                       font=('TkDefaultFont', 9, 'bold'))

        style.configure('Dark.TFrame', 
                       background=self.dark_bg,
                       borderwidth=0,
                       relief='flat')

        style.configure('TLabelFrame', 
                       background=self.dark_bg, 
                       foreground=self.dark_fg, 
                       borderwidth=0, 
                       relief='flat',
                       lightcolor=self.dark_bg,
                       darkcolor=self.dark_bg)
        style.configure('TLabelFrame.Label', background=self.dark_bg, foreground=self.dark_fg)

        style.configure('TLabel', background=self.dark_bg, foreground=self.dark_fg)

        style.configure('Dark.TLabel', 
                       background=self.dark_bg, 
                       foreground=self.dark_fg,
                       font=('TkDefaultFont', 10, 'normal'))

        style.configure('TButton', 
                       background=self.dark_button_bg, 
                       foreground=self.dark_fg,
                       borderwidth=1,
                       focuscolor='none')
        style.map('TButton',
                 background=[('active', '#565656'), ('pressed', '#6a6a6a')])

        style.configure('TEntry', 
                       background=self.dark_entry_bg, 
                       foreground=self.dark_fg,
                       borderwidth=1,
                       insertcolor=self.dark_fg,
                       fieldbackground=self.dark_entry_bg)
        style.map('TEntry',
                 background=[('focus', self.dark_entry_bg), ('active', self.dark_entry_bg)],
                 fieldbackground=[('focus', self.dark_entry_bg), ('active', self.dark_entry_bg)])

        style.configure('TCombobox', 
                       background=self.dark_entry_bg, 
                       foreground=self.dark_fg,
                       borderwidth=1,
                       arrowcolor=self.dark_fg,
                       fieldbackground=self.dark_entry_bg,
                       selectbackground=self.dark_select_bg,
                       insertcolor=self.dark_fg)
        style.map('TCombobox',
                 background=[('focus', self.dark_entry_bg), ('active', self.dark_entry_bg)],
                 fieldbackground=[('focus', self.dark_entry_bg), ('active', self.dark_entry_bg)],
                 selectbackground=[('focus', self.dark_select_bg)],
                 arrowcolor=[('active', self.dark_fg)])

        style.configure('TScale',
                       background=self.dark_bg,
                       troughcolor=self.dark_entry_bg,
                       borderwidth=1,
                       darkcolor=self.dark_button_bg,
                       lightcolor=self.dark_button_bg)

        style.configure('TProgressbar',
                       background='#4a9eff',
                       troughcolor=self.dark_entry_bg,
                       borderwidth=1,
                       lightcolor='#4a9eff',
                       darkcolor='#4a9eff')

        style.configure('TCheckbutton',
                       background=self.dark_bg,
                       foreground=self.dark_fg,
                       focuscolor='none')
        style.map('TCheckbutton',
                 background=[('active', self.dark_bg)])

        style.configure('TScrollbar',
                       background=self.dark_button_bg,
                       darkcolor=self.dark_entry_bg,
                       lightcolor=self.dark_button_bg,
                       troughcolor=self.dark_bg,
                       bordercolor=self.dark_entry_bg,
                       arrowcolor=self.dark_fg)

        root.configure(bg=self.dark_bg)

        root.option_add('*Text.background', self.dark_text_bg)
        root.option_add('*Text.foreground', self.dark_fg)
        root.option_add('*Text.insertBackground', self.dark_fg)
        root.option_add('*Text.selectBackground', self.dark_select_bg)
        root.option_add('*Text.selectForeground', self.dark_fg)

        root.option_add('*Listbox.background', self.dark_entry_bg)
        root.option_add('*Listbox.foreground', self.dark_fg)
        root.option_add('*Listbox.selectBackground', self.dark_select_bg)
        root.option_add('*Listbox.selectForeground', self.dark_fg)

        root.option_add('*Menu.background', self.dark_bg)
        root.option_add('*Menu.foreground', self.dark_fg)
        root.option_add('*Menu.selectColor', self.dark_select_bg)

        root.option_add('*TCombobox*Listbox.background', self.dark_entry_bg)
        root.option_add('*TCombobox*Listbox.foreground', self.dark_fg)
        root.option_add('*TCombobox*Listbox.selectBackground', self.dark_select_bg)
        root.option_add('*TCombobox*Listbox.selectForeground', self.dark_fg)

        root.option_add('*background', self.dark_bg)
        root.option_add('*foreground', self.dark_fg)
        root.option_add('*selectBackground', self.dark_select_bg)
        root.option_add('*selectForeground', self.dark_fg)
        root.option_add('*highlightBackground', self.dark_bg)
        root.option_add('*highlightColor', self.dark_select_bg)
        root.option_add('*troughColor', self.dark_entry_bg)
        root.option_add('*activeBackground', self.dark_select_bg)
        root.option_add('*activeForeground', self.dark_fg)

    def apply_dark_theme_to_window(self, window):
        """Apply dark theme to a specific window or dialog"""
        try:
            window.configure(bg=self.dark_bg)

            def configure_widget(widget):
                widget_class = widget.winfo_class()
                if widget_class == 'Text':
                    widget.configure(bg="#1a1a1a", fg=self.dark_fg, 
                                   insertbackground=self.dark_fg,
                                   selectbackground="#2a2a2a",
                                   selectforeground=self.dark_fg)
                elif widget_class in ['Frame', 'Toplevel']:
                    widget.configure(bg=self.dark_bg)
                elif widget_class == 'Label':
                    widget.configure(bg=self.dark_bg, fg=self.dark_fg)

                for child in widget.winfo_children():
                    configure_widget(child)

            configure_widget(window)
        except:
            pass