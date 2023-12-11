import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from GPBO import *
import pandas as pd


class App(ttk.Frame):
    def __init__(self, parent):
        ttk.Frame.__init__(self)

        self.columnconfigure(index=0, weight=1)
        self.rowconfigure(index=0, weight=0)

        # Create value lists
        self.option_menu_list = ["", "OptionMenu", "Option 1", "Option 2"]
        self.combo_list = ["Combobox", "Editable item 1", "Editable item 2"]
        self.readonly_combo_list = ["Readonly combobox", "Item 1", "Item 2"]

        # Create control variables
        self.var_0 = tk.BooleanVar()
        self.var_1 = tk.BooleanVar(value=True)
        self.var_2 = tk.BooleanVar()
        self.var_3 = tk.IntVar(value=2)
        self.var_4 = tk.StringVar(value=self.option_menu_list[1])
        self.var_5 = tk.DoubleVar(value=75.0)
        self.spinner_var = tk.StringVar()
        self.file_path_var = tk.StringVar()
        self.plot_var = tk.StringVar()


        self.axis_tracker = []
        self.gp = GPToolModel()

        # Create widgets :)
        self.setup_widgets()


    def setup_widgets(self):
        # Create a Frame for the Checkbuttons
        # self.check_frame = ttk.LabelFrame(self, text="Checkbuttons", padding=(20, 10))
        # self.check_frame.grid(
        #     row=0, column=0, padx=(20, 10), pady=(20, 10), sticky="nsew"
        # )
        #
        # # Checkbuttons
        # self.check_1 = ttk.Checkbutton(
        #     self.check_frame, text="Unchecked", variable=self.var_0
        # )
        # self.check_1.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        #
        # self.check_2 = ttk.Checkbutton(
        #     self.check_frame, text="Checked", variable=self.var_1
        # )
        # self.check_2.grid(row=1, column=0, padx=5, pady=10, sticky="nsew")
        #
        # self.check_3 = ttk.Checkbutton(
        #     self.check_frame, text="Third state", variable=self.var_2
        # )
        # self.check_3.state(["alternate"])
        # self.check_3.grid(row=2, column=0, padx=5, pady=10, sticky="nsew")
        #
        # self.check_4 = ttk.Checkbutton(
        #     self.check_frame, text="Disabled", state="disabled"
        # )
        # self.check_4.state(["disabled !alternate"])
        # self.check_4.grid(row=3, column=0, padx=5, pady=10, sticky="nsew")
        #
        # # Separator
        # self.separator = ttk.Separator(self)
        # self.separator.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="ew")
        #
        # # Create a Frame for the Radiobuttons
        # self.radio_frame = ttk.LabelFrame(self, text="Radiobuttons", padding=(20, 10))
        # self.radio_frame.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="nsew")
        #
        # # Radiobuttons
        # self.radio_1 = ttk.Radiobutton(
        #     self.radio_frame, text="Unselected", variable=self.var_3, value=1
        # )
        # self.radio_1.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        # self.radio_2 = ttk.Radiobutton(
        #     self.radio_frame, text="Selected", variable=self.var_3, value=2
        # )
        # self.radio_2.grid(row=1, column=0, padx=5, pady=10, sticky="nsew")
        # self.radio_4 = ttk.Radiobutton(
        #     self.radio_frame, text="Disabled", state="disabled"
        # )
        # self.radio_4.grid(row=3, column=0, padx=5, pady=10, sticky="nsew")
        #
        # # Create a Frame for input widgets
        # self.widgets_frame = ttk.Frame(self, padding=(0, 0, 0, 10))
        # self.widgets_frame.grid(
        #     row=0, column=1, padx=10, pady=(30, 10), sticky="nsew", rowspan=3
        # )
        # self.widgets_frame.columnconfigure(index=0, weight=1)
        #
        # # Entry
        # self.entry = ttk.Entry(self.widgets_frame)
        # self.entry.insert(0, "Entry")
        # self.entry.grid(row=0, column=0, padx=5, pady=(0, 10), sticky="ew")
        #
        # # Spinbox
        # self.spinbox = ttk.Spinbox(self.widgets_frame, from_=0, to=100, increment=0.1)
        # self.spinbox.insert(0, "Spinbox")
        # self.spinbox.grid(row=1, column=0, padx=5, pady=10, sticky="ew")
        #
        # # Combobox
        # self.combobox = ttk.Combobox(self.widgets_frame, values=self.combo_list)
        # self.combobox.current(0)
        # self.combobox.grid(row=2, column=0, padx=5, pady=10, sticky="ew")
        #
        # # Read-only combobox
        # self.readonly_combo = ttk.Combobox(
        #     self.widgets_frame, state="readonly", values=self.readonly_combo_list
        # )
        # self.readonly_combo.current(0)
        # self.readonly_combo.grid(row=3, column=0, padx=5, pady=10, sticky="ew")
        #
        # # Menu for the Menubutton
        # self.menu = tk.Menu(self)
        # self.menu.add_command(label="Menu item 1")
        # self.menu.add_command(label="Menu item 2")
        # self.menu.add_separator()
        # self.menu.add_command(label="Menu item 3")
        # self.menu.add_command(label="Menu item 4")
        #
        # # Menubutton
        # self.menubutton = ttk.Menubutton(
        #     self.widgets_frame, text="Menubutton", menu=self.menu, direction="below"
        # )
        # self.menubutton.grid(row=4, column=0, padx=5, pady=10, sticky="nsew")
        #
        # # OptionMenu
        # self.optionmenu = ttk.OptionMenu(
        #     self.widgets_frame, self.var_4, *self.option_menu_list
        # )
        # self.optionmenu.grid(row=5, column=0, padx=5, pady=10, sticky="nsew")
        #
        # # Button
        # self.button = ttk.Button(self.widgets_frame, text="Button")
        # self.button.grid(row=6, column=0, padx=5, pady=10, sticky="nsew")
        #
        # # Accentbutton
        # self.accentbutton = ttk.Button(
        #     self.widgets_frame, text="Accent button", style="Accent.TButton"
        # )
        # self.accentbutton.grid(row=7, column=0, padx=5, pady=10, sticky="nsew")
        #
        # # Togglebutton
        # self.togglebutton = ttk.Checkbutton(
        #     self.widgets_frame, text="Toggle button", style="Toggle.TButton"
        # )
        # self.togglebutton.grid(row=8, column=0, padx=5, pady=10, sticky="nsew")
        #
        # # Switch
        # self.switch = ttk.Checkbutton(
        #     self.widgets_frame, text="Switch", style="Switch.TCheckbutton"
        # )
        # self.switch.grid(row=9, column=0, padx=5, pady=10, sticky="nsew")
        title_font = ("Arial", 30, "bold")
        title = ttk.Label(self, text="GP BO Tool", font=title_font)
        title.grid(row=0, column=0, sticky="nw")

        # Panedwindow
        self.paned = ttk.PanedWindow(self)
        self.paned.grid(row=1, column=0, pady=(5, 5), sticky="nsew", rowspan=4, columnspan=4)

        # Notebook, pane #2
        self.pane_2 = ttk.Frame(self.paned, padding=5)
        self.paned.add(self.pane_2, weight=1)

        # Notebook, pane #2
        self.notebook = ttk.Notebook(self.pane_2)
        self.notebook.pack(fill="both", expand=False)

        # Tab #1
        self.tab_1 = ttk.Frame(self.notebook)
        # for index in [0, 1]:
        #     self.tab_1.columnconfigure(index=index, weight=1)
        #     self.tab_1.rowconfigure(index=index, weight=1)
        self.notebook.add(self.tab_1, text="Configuration")

        # Label
        self.label = ttk.Label(
            self.tab_1,
            text="Dimension of Design Space",
            justify="left",
            font=("-size", 12, "-weight", "bold"),
        )
        self.label.grid(row=0, column=0, pady=10, sticky="nw")

        self.define_axis = ttk.LabelFrame(self.tab_1, text="Define Axis", padding=(20, 10))
        self.define_axis.grid(row=2, column=0, padx=(20, 10), pady=(20, 10), sticky="nw")

        self.spinner_var.trace("w", self.update_text_widgets)

        self.spinbox = ttk.Spinbox(self.tab_1, from_=1, to=5, increment=1, textvariable=self.spinner_var)
        self.spinbox.insert(0, "1")
        self.spinbox.grid(row=1, column=0, padx=5, pady=5, sticky="nw")

        for index,value in enumerate(["Name", "Lower Limit", "Upper Limit"]):
            self.label = ttk.Label(
                self.define_axis,
                text=value,
                justify="left",
                font=("-size", 12, "-weight", "bold"),
            )
            self.label.grid(row=0, column=index, pady=5, padx=5, sticky="n")

        self.update_text_widgets()

        self.label = ttk.Label(
            self.tab_1,
            text="Upload Data",
            justify="left",
            font=("-size", 12, "-weight", "bold"),
        )
        self.label.grid(row=3, column=0, pady=10, sticky="nw")

        self.browse_button = ttk.Button(self.tab_1, text="Select CSV File", command=self.open_file_browser)
        self.browse_button.grid(row=4, column=0, pady=10, sticky="nw")

        self.file_path_entry = ttk.Entry(self.tab_1, width=100, textvariable=self.file_path_var, state="readonly")
        self.file_path_entry.grid(row=5, column=0, pady=10, sticky="nw")

        self.tree = ttk.Treeview(self.tab_1, height=3)
        self.tree.grid(row=6, column=0, pady=10, sticky="nw")

        self.run = ttk.Button(self.tab_1, text="Run GP BO", command=self.gprun, style="Accent.TButton")
        self.run.grid(row=7, column=0, pady=10, sticky="nw")

        # Tab #2
        self.tab_2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2, text="Result")

        self.label = ttk.Label(
            self.tab_2,
            text="Result",
            justify="left",
            font=("-size", 12, "-weight", "bold"),
        )
        self.label.grid(row=0, column=0, pady=10, sticky="nw")

        self.tree2 = ttk.Treeview(self.tab_2, height=1)
        self.tree2.grid(row=1, column=0, pady=5, sticky="nw")

        self.label = ttk.Label(
            self.tab_2,
            text="Graphing",
            justify="left",
            font=("-size", 12, "-weight", "bold"),
        )
        self.label.grid(row=2, column=0, pady=10, sticky="nw")

        self.plot_var.trace("w", self.plot_graph)

        self.spinbox2 = ttk.Spinbox(self.tab_2, from_=1, to=int(self.spinbox.get()), increment=1, textvariable=self.plot_var)
        self.spinbox2.insert(0, "1")
        self.spinbox2.grid(row=3, column=0, padx=5, pady=5, sticky="nw")



        # Tab #3
        # self.tab_3 = ttk.Frame(self.notebook)
        # self.notebook.add(self.tab_3, text="Tab 3")

        # Sizegrip
        self.sizegrip = ttk.Sizegrip(self)
        self.sizegrip.grid(row=100, column=100, padx=(0, 0), pady=(0, 5))

    def plot_graph(self, *args):
        axis = int(self.spinbox2.get())
        fig = self.gp.graph(axis)
        if fig is not None:

            canvas = FigureCanvasTkAgg(fig, master=self.tab_2)
            canvas.draw()
            canvas.get_tk_widget().grid(row=4, column=0, padx=5, pady=5, sticky="nw")


    def switch_to_tab(self):
        # Switch to the second tab (index 1)
        self.notebook.select(1)  # Replace 1 with the index of the tab you want to switch to

    def gprun(self):
        self.gp.n = int(self.spinbox.get())
        limits = []
        for i in self.axis_tracker:
            limits.append([int(i[1].get("1.0", tk.END).replace("\n", "")), int(i[2].get("1.0", tk.END).replace("\n", ""))])
        self.gp.limits = limits
        te = self.gp.runGPBO()
        print(te)
        self.switch_to_tab()

        headers = []
        for i in self.axis_tracker:
            headers.append(i[0].get("1.0", tk.END).replace("\n", ""))
        #headers.append("y")
        self.tree2["columns"] = headers[1:]
        self.tree2.delete(*self.tree2.get_children())
        self.tree2.heading("#0", text=headers[0])
        self.tree2.insert("", "end", text=te[0], values=tuple(te[1:]))

        print(headers)
        for header in headers[1:]:
            self.tree2.heading(header, text=header)

    def display_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            headers = []
            for i in self.axis_tracker:
                headers.append(i[0].get("1.0", tk.END).replace("\n", ""))
            headers.append("y")
            self.tree["columns"] = headers[1:]
            self.tree.delete(*self.tree.get_children())
            self.tree.heading("#0", text=headers[0])

            data = df.head(5)
            for index, row in data.iterrows():
                self.tree.insert("", "end",text=row.iloc[0], values=tuple(row.iloc[1:]))

            for header in headers[1:]:
                self.tree.heading(header, text=header)

            self.gp.file_name = file_path

        except pd.errors.EmptyDataError:
            print("The CSV file is empty.")
        except Exception as e:
            print(f"Error: {e}")

    def open_file_browser(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(file_path)
            self.display_csv(file_path)

    def update_text_widgets(self, *args):
        num_rows = int(self.spinbox.get())

        if num_rows > len(self.axis_tracker):
            self.add_text_widgets(num_rows - len(self.axis_tracker))
        elif num_rows < len(self.axis_tracker):
            self.remove_text_widgets(len(self.axis_tracker) - num_rows)

    def add_text_widgets(self, num_rows_to_add):
        for i in range(num_rows_to_add):
            row_widgets = []

            for _ in range(3):
                text_widget = tk.Text(self.define_axis, height=1, width=10)
                text_widget.grid(row=i+len(self.axis_tracker)+1, column=_, padx=5, pady=5, sticky="n")
                if _ == 0:
                    text_widget.insert(tk.END, "x{}".format(i+len(self.axis_tracker)+1))
                row_widgets.append(text_widget)
            self.axis_tracker.append(row_widgets)

    def remove_text_widgets(self, num_rows_to_remove):
        for _ in range(num_rows_to_remove):
            row_widgets = self.axis_tracker.pop()
            for widget in row_widgets:
                widget.grid_forget()  # Remove widgets from display


if __name__ == "__main__":
    root = tk.Tk()
    root.title("")

    # Simply set the theme
    root.tk.call("source", "azure.tcl")
    root.tk.call("set_theme", "light")

    app = App(root)
    app.pack(fill="both", expand=True)

    # Set a minsize for the window, and place it in the middle
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    x_cordinate = int((root.winfo_screenwidth() / 2) - (root.winfo_width() / 2))
    y_cordinate = int((root.winfo_screenheight() / 2) - (root.winfo_height() / 2))
    root.geometry("+{}+{}".format(x_cordinate, y_cordinate-20))

    root.mainloop()