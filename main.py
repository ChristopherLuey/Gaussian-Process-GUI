import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import font

import os

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
        self.canvas = None  # Initialize canvas as None

        self.query_tracker = []



        self.axis_tracker = []
        self.gp = GPToolModel()

        # Create widgets :)
        self.setup_widgets()


    def setup_widgets(self):

        title_font = ("Helvetica", 40, "bold")
        minor_font = font.Font(family="Helvetica", size=15)
        bold_font = font.Font(family="Helvetica", size=16, weight="bold")


        title = ttk.Label(self, text="Northwestern IDEAL Statistical Tool", font=title_font, justify="center", foreground="royal blue")
        title.grid(row=0, column=0, sticky="ew")

        style = ttk.Style()
        style.configure("Custom.Treeview", borderwidth=0, relief="flat")

        # Panedwindow
        self.paned = ttk.PanedWindow(self)
        self.paned.grid(row=1, column=0, pady=(5, 5), sticky="nsew", rowspan=4, columnspan=4)

        # Notebook, pane #2
        self.pane_2 = ttk.Frame(self.paned, padding=5)
        self.pane_2.columnconfigure(0, weight=1)
        self.pane_2.rowconfigure(0, weight=1)
        self.paned.add(self.pane_2, weight=1)


        # Notebook, pane #2
        self.notebook = ttk.Notebook(self.pane_2)
        self.notebook.pack(fill="both", expand=True)

        self.tab_1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_1, text="Configuration")

        # Scrollable canvas for Tab 1
        self.tab_1_scroll_canvas = tk.Canvas(self.tab_1, borderwidth=0, height=800)
        self.tab_1_scroll_canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar for the canvas in Tab 1
        self.tab_1_scrollbar = ttk.Scrollbar(self.tab_1, orient="vertical", command=self.tab_1_scroll_canvas.yview)
        self.tab_1_scrollbar.pack(side="right", fill="y")
        self.tab_1_scroll_canvas.configure(yscrollcommand=self.tab_1_scrollbar.set)

        # Frame to contain all widgets in Tab 1
        self.tab_1_frame = ttk.Frame(self.tab_1_scroll_canvas)
        self.tab_1_canvas_window = self.tab_1_scroll_canvas.create_window((0, 0), window=self.tab_1_frame, anchor="nw")

        def configure_scrollregion(event):
            self.tab_1_scroll_canvas.configure(scrollregion=self.tab_1_scroll_canvas.bbox("all"))

        self.tab_1_frame.bind("<Configure>", configure_scrollregion)
        self.tab_1_scroll_canvas.bind("<Configure>", lambda e: self.tab_1_frame.config(width=e.width))

        # Label
        self.label = ttk.Label(
            self.tab_1_frame,
            text="Dimension of Design Space",
            justify="left",
            font=bold_font,
            foreground="royal blue"
        )
        self.label.grid(row=0, column=0, pady=10, sticky="nw")

        self.define_axis = ttk.LabelFrame(self.tab_1_frame, text="Define Axis", padding=(20, 10))
        self.define_axis.grid(row=2, column=0, padx=(20, 10), pady=(20, 10), sticky="nw")
        # Tab #2 setup with a scrollable area
        self.tab_2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2, text="Result")

        # Scrollable canvas for Tab 2
        self.tab_2_scroll_canvas = tk.Canvas(self.tab_2, borderwidth=0)
        self.tab_2_scroll_canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar for the canvas
        self.tab_2_scrollbar = ttk.Scrollbar(self.tab_2, orient="vertical", command=self.tab_2_scroll_canvas.yview)
        self.tab_2_scrollbar.pack(side="right", fill="y")
        self.tab_2_scroll_canvas.configure(yscrollcommand=self.tab_2_scrollbar.set)

        # Frame to contain all widgets in Tab 2
        self.tab_2_frame = ttk.Frame(self.tab_2_scroll_canvas)
        self.tab_2_canvas_window = self.tab_2_scroll_canvas.create_window((0, 0), window=self.tab_2_frame, anchor="nw")

        self.tab_2_frame.bind("<Configure>", lambda e: self.tab_2_scroll_canvas.configure(
            scrollregion=self.tab_2_scroll_canvas.bbox("all")))

        # Now place all widgets of Tab 2 inside self.tab_2_frame
        # Example for placing the graph area:
        self.graph = ttk.LabelFrame(self.tab_2_frame, text="Graphing", padding=(0, 0))
        self.graph.grid(row=2, column=0, padx=(10, 0), pady=(0, 0), sticky="nsew")
        self.graph.columnconfigure(0, weight=1)
        self.graph.rowconfigure(0, weight=1)

        self.holder_frame = ttk.Frame(self.tab_2_frame)
        self.holder_frame.grid(row=1,column=0, sticky="nw")
        self.tree2 = ttk.Treeview(self.holder_frame, height=1, style="Custom.Treeview")
        self.tree2.grid(row=0, column=0, padx=(20, 10), pady=(5, 5), sticky="nw")
        self.query_point = ttk.LabelFrame(self.holder_frame, text="Query Point", padding=(20, 10))
        self.query_point.grid(row=0, column=1, padx=(20, 10), pady=(5, 5), sticky="nw")

        self.query_point_inside = ttk.Frame(self.query_point)
        self.query_point_inside.grid(row=0,column=0, sticky="nw")

        self.input_new = ttk.LabelFrame(self.holder_frame, text="Input New Data", padding=(20, 10))
        self.input_new.grid(row=0, column=2, padx=(20, 10), pady=(5, 5), sticky="nw")

        self.input_new_holder = ttk.Frame(self.input_new)
        self.input_new_holder.grid(row=0, column=0, sticky="nw")

        self.spinner_var.trace("w", self.update_text_widgets)

        self.spinbox = ttk.Spinbox(self.tab_1_frame, from_=1, to=5, increment=1, textvariable=self.spinner_var)
        self.spinbox.insert(0, "1")
        self.spinbox.grid(row=1, column=0, padx=5, pady=5, sticky="nw")

        for index,value in enumerate(["Name", "Lower Limit", "Upper Limit"]):
            self.label = ttk.Label(
                self.define_axis,
                text=value,
                justify="left",
                font=("-family", "Helvetica", "-size", 12, "-weight", "bold"),
            )
            self.label.grid(row=0, column=index, pady=5, padx=5, sticky="nw")

        self.update_text_widgets()

        self.label = ttk.Label(
            self.tab_1_frame,
            text="Upload Data",
            justify="left",
            font=bold_font,
            foreground="royal blue"
        )
        self.label.grid(row=3, column=0, pady=10, sticky="nw")
        self.selectCSV = ttk.LabelFrame(self.tab_1_frame, text="Select CSV", padding=(20, 10))
        self.selectCSV.grid(row=4, column=0, padx=(20, 10), pady=(10, 10), sticky="nw")


        style = ttk.Style()
        style.configure("W.TButton", background="royal blue", foreground="black", font=minor_font)

        self.browse_button = ttk.Button(self.selectCSV, text="Upload", command=self.open_file_browser, style="W.TButton")
        self.browse_button.grid(row=1, column=0, pady=10, padx=0,sticky="nw")

        self.file_path_entry = ttk.Entry(self.selectCSV, width=30, textvariable=self.file_path_var, state="readonly", font=minor_font)
        self.file_path_entry.grid(row=1, column=1, pady=10, padx=5, sticky="nw")

        self.CSVview = ttk.LabelFrame(self.tab_1_frame, text="Preview Data", padding=(20, 10))
        self.CSVview.grid(row=5, column=0, padx=(20, 10), pady=(10, 10), sticky="nw")

        self.tree = ttk.Treeview(self.CSVview, height=3, style="Custom.Treeview")
        self.tree.grid(row=1, column=0, pady=10, sticky="nw")

        style = ttk.Style()
        style.configure('Minor.TButton', font=minor_font, foreground="royal blue")

        self.run = ttk.Button(self.tab_1_frame, text="Run Gaussian Process with Bayesian Optimization", command=self.gprun, style="Minor.TButton")
        self.run.grid(row=6, column=0, padx=15, pady=15, sticky="nw")

        # Tab #2

        self.label = ttk.Label(
            self.tab_2_frame,
            text="Result",
            justify="left",
            font=bold_font,
            foreground="royal blue"
        )
        self.label.grid(row=0, column=0, pady=5, sticky="nw")




        self.plot_var.trace("w", self.plot_graph)

        self.spinbox2 = ttk.Spinbox(self.graph, from_=1, to=int(self.spinbox.get()), increment=1, textvariable=self.plot_var)
        self.spinbox2.insert(0, "1")
        # self.spinbox2.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        self.plot_graph()

        self.query = ttk.Button(self.query_point, text="Query Point", command=self.query_p, style="Accent.TButton")
        self.query.grid(row=1, column=0, pady=10, sticky="nw")

        l = ttk.Label(
            self.input_new_holder,
            text="y=",
            justify="left",
            font=("-size", 12, "-weight", "bold"),
        )
        l.grid(row=0, column=0, pady=5, sticky="n")

        self.yt = tk.Text(self.input_new_holder, height=1, width=5, borderwidth=1, relief="sunken")
        self.yt.grid(row=0, column=1, padx=5, pady=5, sticky="n")

        self.togglebutton = ttk.Checkbutton(
            self.input_new, text="Edit CSV", style="Toggle.TButton"
        )
        self.togglebutton.grid(row=1, column=0, padx=5, pady=1, sticky="n")
        self.new_data = ttk.Button(self.input_new, text="Generate", command=self.input_new_data, style="Accent.TButton")
        self.new_data.grid(row=2, column=0, pady=1, padx=5, sticky="n")

        # Tab #3
        # self.tab_3 = ttk.Frame(self.notebook)
        # self.notebook.add(self.tab_3, text="Tab 3")

        # Sizegrip
        self.sizegrip = ttk.Sizegrip(self)
        self.sizegrip.grid(row=100, column=100, padx=(0, 0), pady=(0, 5))


    def input_new_data(self):
        checked = self.togglebutton.instate(['selected'])
        y = float(self.yt.get("1.0", tk.END).replace("\n", ""))
        print(y)
        print(type(y))
        self.gp.new_data(checked, y)
        self.gprun()


    def query_p(self):
        q = []
        print(self.query_tracker)
        for i in self.query_tracker:
            try:
                q.append(float(i[1].get("1.0", tk.END).replace("\n", "")))
            except:
                print("wrong character input")
                return

        mean, variance = self.gp.query(q)

        if hasattr(self, 'mean_l'):
            self.mean_l.destroy()
        if hasattr(self, 'variance_l'):
            self.variance_l.destroy()

        self.mean_l = ttk.Label(
            self.query_point,
            text="Mean: {}".format(mean),
            justify="left",
            font=("-size", 12, "-weight", "bold"), )
        self.mean_l.grid(row=2, column=0, pady=5, sticky="nw")
        self.variance_l = ttk.Label(
            self.query_point,
            text="Variance: {}".format(variance),
            justify="left",
            font=("-size", 12, "-weight", "bold"), )
        self.variance_l.grid(row=3, column=0, pady=5, sticky="nw")

    def plot_graph(self, *args):
        axis = int(self.spinbox2.get())
        xlabel = self.axis_tracker[-1][0].get("1.0", tk.END).replace("\n", "")
        ylabel = 'y'  # Set y-axis label
        fig1 = self.gp.graph(axis, xlabel, ylabel)  # This should now return a matplotlib figure

        if fig1 is not None:
            # Check if canvas exists before trying to destroy it
            if hasattr(self, 'canvas') and self.canvas is not None:
                self.canvas.get_tk_widget().destroy()  # Destroy previous canvas if exists

            self.canvas = FigureCanvasTkAgg(fig1, master=self.graph)  # Embed in graph_frame
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.grid(row=0, column=0, sticky="nsew")  # Use grid
            self.canvas.draw()

    def update_scrollregion(self, event):
        self.canvas_widget.configure(scrollregion=self.canvas_widget.bbox("all"))


    def switch_to_tab(self):
        # Switch to the second tab (index 1)
        self.notebook.select(1)  # Replace 1 with the index of the tab you want to switch to
        self.plot_graph()

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

        self.tree2["columns"] = headers[1:]
        self.tree2.delete(*self.tree2.get_children())

        # Set the width for the first column (#0)
        self.tree2.column("#0", width=100)  # Adjust the width as needed
        self.tree2.heading("#0", text=headers[0])

        # Configure the width and heading for the rest of the columns
        for index, header in enumerate(headers[1:], start=1):
            col_id = f"#{index}"
            self.tree2.column(col_id, width=100)  # Adjust the width as needed
            self.tree2.heading(col_id, text=header)

        # Insert data
        self.tree2.insert("", "end", text=round(te[0], 4), values=tuple(round(x, 4) for x in te[1:]))

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
            self.gp.read_file()

            self.adjust_tree_columns(self.tree, headers)

        except pd.errors.EmptyDataError:
            print("The CSV file is empty.")
        except Exception as e:
            print(f"Error: {e}")


    def adjust_tree_columns(self, tree, column_headers):
        tree.update_idletasks()  # Update the treeview

        # Measure and adjust the width of additional columns
        for ix, col in enumerate(column_headers, start=1):  # Start from 1 as 0 is the tree column
            tree.column(col, width=font.Font().measure(col.title()))  # Set a base width for the column

            for item in tree.get_children():
                cell_value = tree.item(item, 'values')[ix-1]
                col_w = font.Font().measure(cell_value)
                if tree.column(col, width=None) < col_w:
                    tree.column(col, width=col_w)

    def open_file_browser(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_path_var.set(os.path.basename(file_path))
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
                text_widget = tk.Text(self.define_axis, height=1, width=10, borderwidth=1, relief="sunken")
                text_widget.grid(row=i+len(self.axis_tracker)+1, column=_, padx=5, pady=5, sticky="n")
                if _ == 0:
                    text_widget.insert(tk.END, "x{}".format(i+len(self.axis_tracker)+1))
                row_widgets.append(text_widget)

            self.axis_tracker.append(row_widgets)

            l = ttk.Label(
                self.query_point_inside,
                text=self.axis_tracker[-1][0].get("1.0", tk.END).replace("\n", ""),
                justify="left",
                font=("-size", 12, "-weight", "bold"),)
            l.grid(row=i+len(self.axis_tracker)-1, column=0, pady=5, sticky="n")

            t = tk.Text(self.query_point_inside, height=1, width=5, borderwidth=1, relief="sunken")
            t.grid(row=i + len(self.axis_tracker)-1, column=1, padx=5, pady=5, sticky="n")
            t.insert(tk.END, "0.0")

            self.query_tracker.append([l,t])

    def remove_text_widgets(self, num_rows_to_remove):
        for _ in range(num_rows_to_remove):
            row_widgets = self.axis_tracker.pop()
            w = self.query_tracker.pop()
            for widget in row_widgets:
                widget.grid_forget()  # Remove widgets from display
            for widget in w:
                widget.grid_forget()


if __name__ == "__main__":
    root = tk.Tk()
    # root.geometry('800x600')  # Set initial size of the window
    # root.minsize(800, 600)
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