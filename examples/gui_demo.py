"""
GUI demonstration for CellSegKit.

This script provides a graphical user interface to demonstrate the key features
of CellSegKit, including model selection, input/output directory selection,
export format selection, visualization of results, and mask format conversion.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io

from cellsegkit import SegmenterFactory, run_segmentation, convert_mask_format
from cellsegkit.importer import find_images
from cellsegkit.utils.gpu_utils import check_gpu_availability
from cellsegkit.converter import VALID_FORMATS

class CellSegKitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CellSegKit Demonstration")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)

        # Variables for segmentation
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_type = tk.StringVar(value="cyto")
        self.use_gpu = tk.BooleanVar(value=check_gpu_availability(verbose=False))
        self.export_overlay = tk.BooleanVar(value=True)
        self.export_npy = tk.BooleanVar(value=True)
        self.export_png = tk.BooleanVar(value=True)
        self.export_yolo = tk.BooleanVar(value=True)

        # Variables for mask conversion
        self.mask_path = tk.StringVar()
        self.original_image_path = tk.StringVar()
        self.conversion_output_path = tk.StringVar()
        self.conversion_format = tk.StringVar(value="png")
        self.class_id = tk.IntVar(value=0)

        self.segmenter = None
        self.image_paths = []
        self.current_image_index = 0
        self.segmentation_results = {}

        # Create UI
        self.create_ui()

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook (tabs)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Create segmentation tab
        segmentation_tab = ttk.Frame(notebook, padding=5)
        notebook.add(segmentation_tab, text="Segmentation")

        # Create conversion tab
        conversion_tab = ttk.Frame(notebook, padding=5)
        notebook.add(conversion_tab, text="Mask Conversion")

        # Create segmentation UI
        self.create_segmentation_ui(segmentation_tab)

        # Create conversion UI
        self.create_conversion_ui(conversion_tab)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_segmentation_ui(self, parent):
        # Segmentation tab layout
        segmentation_frame = ttk.Frame(parent)
        segmentation_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel (controls)
        left_panel = ttk.LabelFrame(segmentation_frame, text="Controls", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Model selection
        model_frame = ttk.LabelFrame(left_panel, text="Model Selection", padding=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(model_frame, text="Model Type:").pack(anchor=tk.W)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_type, 
                                   values=["cyto", "nuclei", "cellpose", "cellsam"])
        model_combo.pack(fill=tk.X, pady=(0, 5))

        gpu_check = ttk.Checkbutton(model_frame, text="Use GPU (if available)", 
                                    variable=self.use_gpu)
        gpu_check.pack(anchor=tk.W)

        # Directory selection
        dir_frame = ttk.LabelFrame(left_panel, text="Directories", padding=10)
        dir_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(dir_frame, text="Input Directory:").pack(anchor=tk.W)
        input_entry = ttk.Entry(dir_frame, textvariable=self.input_dir)
        input_entry.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(dir_frame, text="Browse...", command=self.browse_input_dir).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(dir_frame, text="Output Directory:").pack(anchor=tk.W)
        output_entry = ttk.Entry(dir_frame, textvariable=self.output_dir)
        output_entry.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(dir_frame, text="Browse...", command=self.browse_output_dir).pack(anchor=tk.W)

        # Export options
        export_frame = ttk.LabelFrame(left_panel, text="Export Formats", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(export_frame, text="Overlay (visualization)", variable=self.export_overlay).pack(anchor=tk.W)
        ttk.Checkbutton(export_frame, text="NumPy (data analysis)", variable=self.export_npy).pack(anchor=tk.W)
        ttk.Checkbutton(export_frame, text="PNG (indexed masks)", variable=self.export_png).pack(anchor=tk.W)
        ttk.Checkbutton(export_frame, text="YOLO (object detection)", variable=self.export_yolo).pack(anchor=tk.W)

        # Action buttons
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(action_frame, text="Load Images", command=self.load_images).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Run Segmentation", command=self.run_segmentation).pack(fill=tk.X)

        # Right panel (visualization)
        right_panel = ttk.LabelFrame(segmentation_frame, text="Visualization", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Image navigation
        nav_frame = ttk.Frame(right_panel)
        nav_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT)
        self.image_label = ttk.Label(nav_frame, text="No images loaded")
        self.image_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT)

        # Image display
        self.display_frame = ttk.Frame(right_panel)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

    def browse_input_dir(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir.set(directory)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def load_images(self):
        input_dir = self.input_dir.get()
        if not input_dir:
            messagebox.showerror("Error", "Please select an input directory")
            return

        self.status_var.set("Loading images...")
        self.root.update_idletasks()

        try:
            self.image_paths = find_images(input_dir)
            if not self.image_paths:
                messagebox.showinfo("Info", "No images found in the selected directory")
                self.status_var.set("No images found")
                return

            self.current_image_index = 0
            self.segmentation_results = {}
            self.display_current_image()
            self.status_var.set(f"Loaded {len(self.image_paths)} images")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")
            self.status_var.set("Error loading images")

    def run_segmentation(self):
        if not self.image_paths:
            messagebox.showerror("Error", "Please load images first")
            return

        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()

        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return

        # Get export formats
        export_formats = []
        if self.export_overlay.get():
            export_formats.append("overlay")
        if self.export_npy.get():
            export_formats.append("npy")
        if self.export_png.get():
            export_formats.append("png")
        if self.export_yolo.get():
            export_formats.append("yolo")

        if not export_formats:
            messagebox.showerror("Error", "Please select at least one export format")
            return

        # Create segmenter
        try:
            self.status_var.set("Creating segmenter...")
            self.root.update_idletasks()

            self.segmenter = SegmenterFactory.create(
                model_type=self.model_type.get(),
                use_gpu=self.use_gpu.get()
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create segmenter: {str(e)}")
            self.status_var.set("Error creating segmenter")
            return

        # Run segmentation in a separate thread
        self.status_var.set("Running segmentation...")
        threading.Thread(target=self._run_segmentation_thread, 
                         args=(input_dir, output_dir, export_formats)).start()

    def _run_segmentation_thread(self, input_dir, output_dir, export_formats):
        try:
            run_segmentation(
                segmenter=self.segmenter,
                input_dir=input_dir,
                output_dir=output_dir,
                export_formats=export_formats
            )

            # Update UI in the main thread
            self.root.after(0, self._segmentation_complete, output_dir)
        except Exception as e:
            # Update UI in the main thread
            self.root.after(0, self._segmentation_error, str(e))

    def _segmentation_complete(self, output_dir):
        self.status_var.set("Segmentation complete")
        messagebox.showinfo("Success", f"Segmentation complete. Results saved to {output_dir}")

        # Try to load overlay results for visualization
        try:
            self.load_segmentation_results(output_dir)
        except:
            pass

    def _segmentation_error(self, error_msg):
        self.status_var.set("Segmentation failed")
        messagebox.showerror("Error", f"Segmentation failed: {error_msg}")

    def load_segmentation_results(self, output_dir):
        self.segmentation_results = {}

        # Check for overlay results
        if self.export_overlay.get():
            overlay_dir = os.path.join(output_dir, "overlay")
            if os.path.exists(overlay_dir):
                for image_path in self.image_paths:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    overlay_path = os.path.join(overlay_dir, f"{base_name}_overlay.png")
                    if os.path.exists(overlay_path):
                        self.segmentation_results[image_path] = overlay_path

        # Update display
        self.display_current_image()

    def display_current_image(self):
        if not self.image_paths:
            return

        # Clear previous display
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        current_path = self.image_paths[self.current_image_index]
        self.image_label.config(text=f"Image {self.current_image_index + 1} of {len(self.image_paths)}")

        # Create figure for matplotlib
        fig = plt.Figure(figsize=(10, 6), dpi=100)

        # Original image
        try:
            img = Image.open(current_path)
            ax1 = fig.add_subplot(121)
            ax1.imshow(np.array(img))
            ax1.set_title("Original Image")
            ax1.axis('off')

            # Segmentation result (if available)
            if current_path in self.segmentation_results:
                overlay_path = self.segmentation_results[current_path]
                overlay_img = Image.open(overlay_path)
                ax2 = fig.add_subplot(122)
                ax2.imshow(np.array(overlay_img))
                ax2.set_title("Segmentation Result")
                ax2.axis('off')
            else:
                ax2 = fig.add_subplot(122)
                ax2.text(0.5, 0.5, "No segmentation result available", 
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("Segmentation Result")
                ax2.axis('off')

            # Add to display frame
            canvas = FigureCanvasTkAgg(fig, master=self.display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            error_label = ttk.Label(self.display_frame, 
                                   text=f"Error displaying image: {str(e)}")
            error_label.pack(fill=tk.BOTH, expand=True)

    def prev_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.display_current_image()

    def next_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.display_current_image()

    def create_conversion_ui(self, parent):
        # Conversion tab layout
        conversion_frame = ttk.Frame(parent)
        conversion_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel (controls)
        left_panel = ttk.LabelFrame(conversion_frame, text="Mask Conversion Controls", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Mask file selection
        mask_frame = ttk.LabelFrame(left_panel, text="Mask File", padding=10)
        mask_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mask_frame, text="Mask Path (.npy or .png):").pack(anchor=tk.W)
        mask_entry = ttk.Entry(mask_frame, textvariable=self.mask_path)
        mask_entry.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(mask_frame, text="Browse...", command=self.browse_mask_file).pack(anchor=tk.W)

        # Original image selection (for overlay and YOLO formats)
        image_frame = ttk.LabelFrame(left_panel, text="Original Image", padding=10)
        image_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(image_frame, text="Original Image Path:").pack(anchor=tk.W)
        ttk.Label(image_frame, text="(Required for overlay and YOLO formats)", font=("", 8)).pack(anchor=tk.W)
        image_entry = ttk.Entry(image_frame, textvariable=self.original_image_path)
        image_entry.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(image_frame, text="Browse...", command=self.browse_original_image).pack(anchor=tk.W)

        # Output format selection
        format_frame = ttk.LabelFrame(left_panel, text="Output Format", padding=10)
        format_frame.pack(fill=tk.X, pady=(0, 10))

        format_values = list(VALID_FORMATS)
        format_combo = ttk.Combobox(format_frame, textvariable=self.conversion_format, values=format_values)
        format_combo.pack(fill=tk.X, pady=(0, 5))

        # YOLO class ID (only for YOLO format)
        class_frame = ttk.LabelFrame(left_panel, text="YOLO Class ID", padding=10)
        class_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(class_frame, text="Class ID (for YOLO format only):").pack(anchor=tk.W)
        class_entry = ttk.Entry(class_frame, textvariable=self.class_id)
        class_entry.pack(fill=tk.X, pady=(0, 5))

        # Output file selection
        output_frame = ttk.LabelFrame(left_panel, text="Output File", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(output_frame, text="Output Path:").pack(anchor=tk.W)
        output_entry = ttk.Entry(output_frame, textvariable=self.conversion_output_path)
        output_entry.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(output_frame, text="Browse...", command=self.browse_conversion_output).pack(anchor=tk.W)

        # Convert button
        convert_button = ttk.Button(left_panel, text="Convert Mask", command=self.convert_mask)
        convert_button.pack(fill=tk.X, pady=(10, 0))

        # Right panel (preview)
        right_panel = ttk.LabelFrame(conversion_frame, text="Preview", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Preview display
        self.conversion_display_frame = ttk.Frame(right_panel)
        self.conversion_display_frame.pack(fill=tk.BOTH, expand=True)

        # Initial message
        message = ttk.Label(self.conversion_display_frame, 
                           text="Select a mask file and output format to preview conversion")
        message.pack(fill=tk.BOTH, expand=True)

    def browse_mask_file(self):
        file_types = [
            ("Mask files", "*.npy;*.png"),
            ("NumPy files", "*.npy"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="Select Mask File",
            filetypes=file_types
        )
        if file_path:
            self.mask_path.set(file_path)
            # Try to suggest an output path based on the mask path
            self._suggest_conversion_output_path()

    def browse_original_image(self):
        file_types = [
            ("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("TIFF files", "*.tif;*.tiff"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="Select Original Image",
            filetypes=file_types
        )
        if file_path:
            self.original_image_path.set(file_path)

    def browse_conversion_output(self):
        # Determine file extension based on selected format
        format_type = self.conversion_format.get()
        if format_type == "npy":
            file_types = [("NumPy files", "*.npy")]
            default_ext = ".npy"
        elif format_type == "png":
            file_types = [("PNG files", "*.png")]
            default_ext = ".png"
        elif format_type == "overlay":
            file_types = [("PNG files", "*.png")]
            default_ext = ".png"
        elif format_type == "yolo":
            file_types = [("Text files", "*.txt")]
            default_ext = ".txt"
        else:
            file_types = [("All files", "*.*")]
            default_ext = ""

        file_path = filedialog.asksaveasfilename(
            title="Save Output As",
            filetypes=file_types,
            defaultextension=default_ext
        )
        if file_path:
            self.conversion_output_path.set(file_path)

    def _suggest_conversion_output_path(self):
        mask_path = self.mask_path.get()
        if not mask_path:
            return

        # Get directory, filename, and extension
        directory = os.path.dirname(mask_path)
        filename = os.path.splitext(os.path.basename(mask_path))[0]

        # Determine output extension based on selected format
        format_type = self.conversion_format.get()
        if format_type == "npy":
            ext = ".npy"
        elif format_type == "png" or format_type == "overlay":
            ext = ".png"
        elif format_type == "yolo":
            ext = ".txt"
        else:
            ext = ".out"

        # Create suggested output path
        output_path = os.path.join(directory, f"{filename}_converted_{format_type}{ext}")
        self.conversion_output_path.set(output_path)

    def convert_mask(self):
        # Validate inputs
        mask_path = self.mask_path.get()
        output_format = self.conversion_format.get()
        output_path = self.conversion_output_path.get()

        if not mask_path:
            messagebox.showerror("Error", "Please select a mask file")
            return

        if not output_format:
            messagebox.showerror("Error", "Please select an output format")
            return

        if not output_path:
            messagebox.showerror("Error", "Please specify an output path")
            return

        # Check if original image is required
        original_image_path = None
        if output_format in ["overlay", "yolo"]:
            original_image_path = self.original_image_path.get()
            if not original_image_path:
                messagebox.showerror("Error", f"Original image is required for {output_format} format")
                return

        # Get class ID for YOLO format
        class_id = 0
        if output_format == "yolo":
            try:
                class_id = self.class_id.get()
            except:
                # Use default class ID (0) if invalid
                pass

        # Run conversion in a separate thread
        self.status_var.set(f"Converting mask to {output_format} format...")
        threading.Thread(target=self._convert_mask_thread, 
                         args=(mask_path, output_format, output_path, original_image_path, class_id)).start()

    def _convert_mask_thread(self, mask_path, output_format, output_path, original_image_path, class_id):
        try:
            # Run the conversion
            convert_mask_format(
                mask_path=mask_path,
                output_format=output_format,
                output_path=output_path,
                original_image_path=original_image_path,
                class_id=class_id
            )

            # Update UI in the main thread
            self.root.after(0, self._conversion_complete, output_path)
        except Exception as e:
            # Update UI in the main thread
            self.root.after(0, self._conversion_error, str(e))

    def _conversion_complete(self, output_path):
        self.status_var.set("Conversion complete")
        messagebox.showinfo("Success", f"Mask conversion complete. Result saved to {output_path}")

        # Try to display the result
        try:
            self._display_conversion_result(output_path)
        except:
            pass

    def _conversion_error(self, error_msg):
        self.status_var.set("Conversion failed")
        messagebox.showerror("Error", f"Mask conversion failed: {error_msg}")

    def _display_conversion_result(self, output_path):
        # Clear previous display
        for widget in self.conversion_display_frame.winfo_children():
            widget.destroy()

        # Create figure for matplotlib
        fig = plt.Figure(figsize=(8, 6), dpi=100)

        # Display based on file type
        if output_path.lower().endswith((".png", ".jpg", ".jpeg")):
            # Display image
            try:
                img = Image.open(output_path)
                ax = fig.add_subplot(111)
                ax.imshow(np.array(img))
                ax.set_title("Conversion Result")
                ax.axis('off')

                # Add to display frame
                canvas = FigureCanvasTkAgg(fig, master=self.conversion_display_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            except Exception as e:
                error_label = ttk.Label(self.conversion_display_frame, 
                                       text=f"Error displaying result: {str(e)}")
                error_label.pack(fill=tk.BOTH, expand=True)
        else:
            # For non-image files, just show a success message
            message = ttk.Label(self.conversion_display_frame, 
                               text=f"Conversion successful!\nOutput saved to:\n{output_path}")
            message.pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = CellSegKitGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
