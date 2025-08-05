import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from ORCAvibtools import vibrational_viewer

def list_out_files_and_view(base_dir='.', x_min=1000, x_max=1800, fwhm=15):
    # Step 1: Find all .out files
    out_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.out'):
                full_path = os.path.join(root, file)
                out_files.append(full_path)

    if not out_files:
        print("No .out files found.")
        return None

    # Step 2: Create dropdown and load button
    dropdown = widgets.Dropdown(
        options=out_files,
        description='Select .out:',
        layout=widgets.Layout(width='80%')
    )

    load_button = widgets.Button(
        description='Load',
        button_style='primary'
    )

    output = widgets.Output()

    # Step 3: Define button click behavior
    def on_button_click(b):
        selected_file = dropdown.value
        with output:
            clear_output(wait=True)
            print(f"Loading: {selected_file}")
            try:
                vibrational_viewer(selected_file, x_min=x_min, x_max=x_max, fwhm=fwhm)
            except Exception as e:
                print(f"Error loading vibrational viewer: {e}")

    load_button.on_click(on_button_click)

    # Step 4: Display everything
    ui = widgets.VBox([dropdown, load_button, output])
    display(ui)

    return dropdown, load_button
