import py3Dmol
from IPython.display import display, clear_output
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np

def vibrational_viewer(
    filepath,
    mode_index=0,
    amplitude=1.0,
    interval=40,
    fwhm=10.0,
    x_min=None,
    x_max=None,
    zoom_adj=0.0
):
    parser=ORCAParser(filepath)
    # === Load data once ===
    df_coords = parser.extract_final_coordinates()
    x, y, z, atoms = parser.dataframe_to_coords_and_atoms(df_coords)
    coords = np.column_stack((x, y, z))

    df_vibdata = parser.parse_orca_vibrational_data()
    new_df = parser.expand_vibrations_to_xyz(df_vibdata)

    df = parser.parse_orca_ir_spectrum()
    frequencies = df["Frequency (cm⁻¹)"].round(2).values
    intensities = df["Intensity (km/mol)"].values

    if x_min is None:
        x_min = 0
    if x_max is None:
        x_max = frequencies.max() * 1.05

    # === State ===
    state = {
        'mode_index': mode_index,
        'amplitude': amplitude,
        'interval': interval,
        'fwhm': fwhm,
        'x_min': x_min,
        'x_max': x_max
    }

    # === Widgets ===
    options = [f"{i}: {freq} cm⁻¹" for i, freq in enumerate(frequencies)]
    mode_list_label = widgets.Label("Mode (cm⁻¹):")
    mode_list = widgets.Select(
        options=options,
        rows=25,
        layout=widgets.Layout(width='250px')
    )

    amplitude_slider = widgets.FloatSlider(
        value=amplitude,
        min=0.1,
        max=10.0,
        step=0.1,
        description='Amplitude:',
        continuous_update=False,
        readout_format='.1f',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    interval_slider = widgets.IntSlider(
        value=interval,
        min=1,
        max=100,
        step=1,
        description='Interval (ms):',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    update_button = widgets.Button(
        description="Update Animation",
        layout=widgets.Layout(width='300px')
    )

    fwhm_input = widgets.FloatText(
        value=fwhm,
        description='FWHM:',
        layout=widgets.Layout(width='150px'),
        style={'description_width': '50px'}
    )

    xrange_min = widgets.FloatText(
        value=x_min,
        description='X min:',
        layout=widgets.Layout(width='150px'),
        style={'description_width': '50px'}
    )

    xrange_max = widgets.FloatText(
        value=x_max,
        description='X max:',
        layout=widgets.Layout(width='150px'),
        style={'description_width': '50px'}
    )

    out_view = widgets.Output(layout=widgets.Layout(border='3px solid gray'))
    out_plot = widgets.Output(layout=widgets.Layout(border='3px solid gray'))
    out_info = widgets.Output(layout=widgets.Layout(border='3px solid gray', padding='5px'))

    # === Functions ===
    def gaussian_broadening(freqs, intensities, fwhm, freq_range):
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        broadened = np.zeros_like(freq_range)
        for f_i, I_i in zip(freqs, intensities):
            broadened += I_i * np.exp(-((freq_range - f_i) ** 2) / (2 * sigma**2))
        return broadened

    def update_view():
        mode_index = state['mode_index']
        amplitude = state['amplitude']
        interval = state['interval']
        displacements = parser.get_mode_array(new_df, mode_index=mode_index)
        xyz = f"{len(atoms)}\n\n"
        for atom, pos, disp in zip(atoms, coords, displacements.T):
            xyz += f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {disp[0]:.6f} {disp[1]:.6f} {disp[2]:.6f}\n"
        with out_view:
            clear_output(wait=True)
            view = py3Dmol.view(width=500, height=500)
            view.addModel(xyz, 'xyz', {'vibrate': {'frames': 10, 'amplitude': amplitude}})
            view.setStyle({'stick': {}})
            view.setBackgroundColor('0xffffff')
            view.animate({'loop': 'backAndForth', 'interval': interval})
            view.zoomTo()
            if not zoom_adj==0:
                view.zoom({'factor':zoom_adj})
                view.render()
            view.show()

    def update_plot():
        with out_plot:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x_min, x_max = state['x_min'], state['x_max']
            mask = (frequencies >= x_min) & (frequencies <= x_max)
            freqs_plot = frequencies[mask]
            intensities_plot = intensities[mask]
            bars = ax.bar(freqs_plot, intensities_plot, width=5, edgecolor='black', alpha=0.7)
            selected_mode = state['mode_index']
            inside_idx = np.where(mask)[0]
            if selected_mode in inside_idx:
                selected_idx = np.where(inside_idx == selected_mode)[0][0]
                bars[selected_idx].set_color('red')
            freq_range = np.linspace(x_min, x_max, 1000)
            broadened = gaussian_broadening(freqs_plot, intensities_plot, state['fwhm'], freq_range)
            ax.plot(freq_range, broadened, color='black', lw=2, label='Gaussian broadening')
            ax.set_xlim(x_min, x_max)
    
            # Rescale y-axis considering broadened max and raw intensities max:
            max_intensity = max(intensities_plot.max() if intensities_plot.size > 0 else 0, broadened.max())
            ax.set_ylim(0, max_intensity * 1.2 if max_intensity > 0 else 1)
    
            ax.set_xlabel('Frequency (cm⁻¹)')
            ax.set_ylabel('Intensity (km/mol)')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            plt.tight_layout()
            display(fig)
            plt.close(fig)

    def update_info():
        with out_info:
            clear_output()
            idx = state['mode_index']
            freq = frequencies[idx]
            inten = intensities[idx]
            print(f"Selected Mode: {idx}")
            print(f"Frequency: {freq:.2f} cm⁻¹")
            print(f"Intensity: {inten:.3f} km/mol")

    # === Observers ===
    def on_mode_change(change):
        if change['name'] == 'value' and change['new'] is not None:
            idx = int(change['new'].split(":")[0])
            state['mode_index'] = idx
            update_view()
            update_plot()
            update_info()

    def on_update_button_click(_):
        state['amplitude'] = amplitude_slider.value
        state['interval'] = interval_slider.value
        update_view()

    def on_fwhm_change(change):
        try:
            val = float(change['new'])
            state['fwhm'] = val
            update_plot()
        except ValueError:
            pass

    def on_xmin_change(change):
        try:
            val = float(change['new'])
            if val < state['x_max']:
                state['x_min'] = val
                update_plot()
        except ValueError:
            pass

    def on_xmax_change(change):
        try:
            val = float(change['new'])
            if val > state['x_min']:
                state['x_max'] = val
                update_plot()
        except ValueError:
            pass

    # === Attach Observers ===
    mode_list.observe(on_mode_change, names='value')
    update_button.on_click(on_update_button_click)
    fwhm_input.observe(on_fwhm_change, names='value')
    xrange_min.observe(on_xmin_change, names='value')
    xrange_max.observe(on_xmax_change, names='value')

    # === Layout ===
    viewer_column = widgets.VBox(
    [out_view, amplitude_slider, interval_slider, update_button],
    layout=widgets.Layout(width='35%')
)
    mode_column = widgets.VBox(
    [mode_list_label, mode_list],
    layout=widgets.Layout(width='15%', margin='10px')
)
    plot_controls_row = widgets.HBox([fwhm_input, xrange_min, xrange_max])
    plot_column = widgets.VBox(
    [out_plot, plot_controls_row, out_info],
    layout=widgets.Layout(width='50%')
)
    
    ui = widgets.HBox(
    [viewer_column, mode_column, plot_column],
    layout=widgets.Layout(
        width='100%',
        flex_flow='row nowrap',  # prevent wrapping
        align_items='stretch',
        overflow_x='auto'
    )
)

    display(ui)
    mode_list.value = options[mode_index]  # trigger initial update




class ORCAParser:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.lines = f.readlines()

    def extract_final_coordinates(self):
        block_indices = [i for i, line in enumerate(self.lines) if "CARTESIAN COORDINATES (ANGSTROEM)" in line]
        if not block_indices:
            raise ValueError("No 'CARTESIAN COORDINATES (ANGSTROEM)' block found in file.")
        
        start_idx = block_indices[-1] + 2
        coords = []

        for line in self.lines[start_idx:]:
            if not line.strip() or re.match(r'\s*-+\s*', line):
                break
            parts = line.split()
            if len(parts) != 4:
                continue
            atom, x, y, z = parts
            coords.append([atom, float(x), float(y), float(z)])

        return pd.DataFrame(coords, columns=["Atom", "X", "Y", "Z"])

    def dataframe_to_coords_and_atoms(self, df):
        x = df["X"].to_numpy()
        y = df["Y"].to_numpy()
        z = df["Z"].to_numpy()
        atoms = df["Atom"].tolist()
        return x, y, z, atoms

    def parse_frequencies(self):
        frequencies = []
        freq_pattern = re.compile(r'^\s*\d+:\s+([0-9.]+)\s+cm\*\*-1')
        parsing = False
        for i, line in enumerate(self.lines):
            match = freq_pattern.match(line)
            if match:
                if not parsing:
                    parsing = True
                freq = float(match.group(1))
                frequencies.append(freq)
            elif parsing:
                break
        if not frequencies:
            print("Warning: No vibrational frequencies found.")
        return frequencies

    def parse_normal_modes(self):
        mode_data_blocks = []
        current_cols = []
        current_block = []
        reading_block = False

        for line in self.lines:
            if "NORMAL MODES" in line:
                reading_block = True
                continue

            if reading_block:
                col_header_match = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
                if col_header_match:
                    if current_block:
                        mode_data_blocks.append((current_cols, current_block))
                        current_block = []
                    current_cols = [int(i) for i in line.strip().split()]
                    continue

                row_match = re.match(r'\s*(\d+)\s+((?:-?\d+\.\d+\s+)+)', line)
                if row_match:
                    index = int(row_match.group(1))
                    values = [float(val) for val in row_match.group(2).split()]
                    current_block.append((index, values))
                    continue

                if line.strip() == "" and current_block:
                    mode_data_blocks.append((current_cols, current_block))
                    current_block = []

        matrix_dict = {}
        for cols, rows in mode_data_blocks:
            for row_idx, values in rows:
                if row_idx not in matrix_dict:
                    matrix_dict[row_idx] = {}
                for col_idx, val in zip(cols, values):
                    matrix_dict[row_idx][col_idx] = val

        df = pd.DataFrame.from_dict(matrix_dict, orient='index')
        df = df.sort_index(axis=0).sort_index(axis=1)
        df.index.name = 'Atom Coordinate Index'
        return df

    def combine_modes_with_frequencies(self, normal_modes_df, frequencies_list):
        col_map = {i: freq for i, freq in enumerate(frequencies_list)}
        return normal_modes_df.rename(columns=col_map)

    def parse_orca_vibrational_data(self):
        freqs = self.parse_frequencies()
        modes = self.parse_normal_modes()
        return self.combine_modes_with_frequencies(modes, freqs)

    def expand_vibrations_to_xyz(self, df):
        df = df.iloc[:, 6:]
        if df.shape[0] % 3 != 0:
            raise ValueError("Row count must be divisible by 3 (x, y, z rows per atom).")

        n_atoms = df.shape[0] // 3
        new_rows = []

        for i in range(n_atoms):
            x_row = df.iloc[i * 3]
            y_row = df.iloc[i * 3 + 1]
            z_row = df.iloc[i * 3 + 2]
            atom_data = {}

            for col in df.columns:
                atom_data[f"{col}_x"] = x_row[col]
                atom_data[f"{col}_y"] = y_row[col]
                atom_data[f"{col}_z"] = z_row[col]

            new_rows.append(atom_data)

        return pd.DataFrame(new_rows)

    def parse_orca_ir_spectrum(self):
        start_index = None
        for i, line in enumerate(self.lines):
            if "IR SPECTRUM" in line:
                start_index = i
                break

        if start_index is None:
            raise ValueError("IR SPECTRUM section not found in the file.")

        data_lines = []
        for line in self.lines[start_index:]:
            if re.match(r'\s*\d+:', line):
                data_lines.append(line)
            elif data_lines:
                break

        data = []
        for line in data_lines:
            match = re.match(
                r'\s*(\d+):\s+([\d.]+)\s+([\d.Ee+-]+)\s+([\d.]+)\s+([\d.]+)\s+\(([^)]+)\)',
                line
            )
            if match:
                mode = int(match.group(1))
                freq = float(match.group(2))
                eps = float(match.group(3))
                intensity = float(match.group(4))
                t2 = float(match.group(5))
                tx, ty, tz = map(float, match.group(6).split())
                data.append([mode, freq, eps, intensity, t2, tx, ty, tz])

        df = pd.DataFrame(data, columns=[
            'Mode', 'Frequency (cm⁻¹)', 'Eps (L/mol*cm)',
            'Intensity (km/mol)', 'T^2 (a.u.)', 'TX', 'TY', 'TZ'
        ])
        return df

    def get_mode_array(self, expanded_df, mode_index, start_column=6):
        col_start = start_column + mode_index * 3
        cols = expanded_df.columns[col_start:col_start + 3]
        return expanded_df[cols].to_numpy().T
