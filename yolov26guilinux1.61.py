import os
import sys
import time
import threading
import queue
import logging
import json
import yaml
import re

# GUI and Visualization Libraries (MUST BE IMPORTED BEFORE AI LIBRARIES ON LINUX)
import customtkinter as ctk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")  
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import pandas as pd
from PIL import Image

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER 

# Global Theme Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class QueueRedirector:
    """Thread-safe redirector that strips terminal ANSI codes and passes text to Tkinter."""
    def __init__(self, log_queue, original_stream):
        self.log_queue = log_queue
        self.original_stream = original_stream
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, str_val):
        self.original_stream.write(str_val)
        self.original_stream.flush()
        # Clean terminal color codes before sending to Tkinter Textbox
        clean_text = self.ansi_escape.sub('', str_val)
        self.log_queue.put(clean_text)

    def flush(self):
        self.original_stream.flush()

    def isatty(self):
        return hasattr(self.original_stream, 'isatty') and self.original_stream.isatty()

class CTkToolTip:
    """Lightweight, shrink-wrapped tooltip widget that automatically dismisses."""
    def __init__(self, widget, text, delay=250):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self._schedule_id = None

        self.widget.bind("<Enter>", self.schedule_tooltip, add="+")
        self.widget.bind("<Leave>", self.hide_tooltip, add="+")
        self.widget.bind("<ButtonPress>", self.hide_tooltip, add="+")
        self.widget.bind("<Unmap>", self.hide_tooltip, add="+")
        self.widget.bind("<Destroy>", self.hide_tooltip, add="+")

    def schedule_tooltip(self, event=None):
        self.cancel_scheduled()
        self._schedule_id = self.widget.after(self.delay, self.show_tooltip)

    def cancel_scheduled(self):
        if self._schedule_id:
            try:
                self.widget.after_cancel(self._schedule_id)
            except Exception:
                pass
            self._schedule_id = None

    def show_tooltip(self, event=None):
        self._schedule_id = None
        if self.tooltip_window or not self.text:
            return
        if not self.widget.winfo_exists() or not self.widget.winfo_viewable():
            return

        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + 10

        try:
            import tkinter as tk
            self.tooltip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.attributes("-topmost", True)

            card = tk.Frame(tw, background="#334155", padx=1, pady=1)
            card.pack(fill="both", expand=True)

            lbl = tk.Label(
                card,
                text=self.text,
                justify="left",
                background="#0f172a",
                foreground="#f8fafc",
                font=("DejaVu Sans", 9),
                padx=10,
                pady=6,
                wraplength=300
            )
            lbl.pack()

            tw.update_idletasks()
            tw.wm_geometry(f"+{x}+{y}")

            tw.bind("<Enter>", self.hide_tooltip)
            tw.bind("<ButtonPress>", self.hide_tooltip)
        except Exception:
            self.hide_tooltip()

    def hide_tooltip(self, event=None):
        self.cancel_scheduled()
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
            self.tooltip_window = None

class YOLO26App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Linux Display & HiDPI Vector Smoothing ---
        ctk.set_widget_scaling(1.25)
        ctk.set_window_scaling(1.25)
        try:
            self.tk.call('tk', 'scaling', 1.5)
        except Exception:
            pass

        self.title("YOLO26 Control Center & Industrial Deployment Suite")
        self.geometry("1240x980")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Training State & Analytics Variables
        self.start_time = 0
        self.total_epochs = 100
        self.last_run_dir = ""
        self.log_queue = queue.Queue()
        
        # Class Filtering Variables
        self.class_vars = {}
        self.class_names = {}
        
        # Pause and Stop flags
        self.pause_event = threading.Event()
        self.pause_event.set()  
        self.stop_requested = False

        # Build Main UI Shell
        self.create_header()

        self.tabview = ctk.CTkTabview(self, width=1200, height=880)
        self.tabview.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")

        # Configured Tabs
        self.tab_dataset = self.tabview.add(" 📁 Dataset Manager ")
        self.tab_classes = self.tabview.add(" Classes ")
        self.tab_train = self.tabview.add(" Model Training ")
        self.tab_adv = self.tabview.add(" Advance Hyperparameters ")
        self.tab_aug = self.tabview.add(" Augmentation ")
        self.tab_metrics = self.tabview.add(" Live mAP Metrics ")
        self.tab_predict = self.tabview.add(" Inference & Testing ")
        self.tab_export = self.tabview.add(" Model Export ")

        self.setup_dataset_manager_tab()
        self.setup_classes_tab()
        self.setup_training_tab()
        self.setup_advanced_hyperparams_tab()
        self.setup_augmentation_tab()
        self.setup_metrics_tab()
        self.setup_prediction_tab()
        self.setup_export_tab()
        
        self.poll_log_queue()

    def create_header(self):
        header_frame = ctk.CTkFrame(self, corner_radius=0, height=52)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)

        title_label = ctk.CTkLabel(
            header_frame, text="YOLO26 Industrial Trainer & Multi-Stage Deployment Suite", font=ctk.CTkFont(size=19, weight="bold")
        )
        title_label.pack(side="left", padx=20, pady=10)

        recipe_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        recipe_frame.pack(side="left", padx=15)
        
        ctk.CTkButton(recipe_frame, text="Load Recipe", width=105, command=self.load_recipe, fg_color="#475569", hover_color="#334155").pack(side="left", padx=4)
        ctk.CTkButton(recipe_frame, text="Save Recipe", width=105, command=self.save_recipe, fg_color="#475569", hover_color="#334155").pack(side="left", padx=4)

        gpu_ok = torch.cuda.is_available()
        gpu_text = f"GPU: {torch.cuda.get_device_name(0)}" if gpu_ok else "CPU Mode Active"
        gpu_color = "#2ef072" if gpu_ok else "#ff4d4d"

        status_label = ctk.CTkLabel(
            header_frame, text=gpu_text, text_color=gpu_color, font=ctk.CTkFont(size=13, weight="bold")
        )
        status_label.pack(side="right", padx=20, pady=10)

    def poll_log_queue(self):
        while not self.log_queue.empty():
            try:
                text = self.log_queue.get_nowait()
                self.console_text.configure(state="normal")
                
                if '\r' in text:
                    text = text.split('\r')[-1]
                    self.console_text.delete("end-1c linestart", "end-1c lineend")
                    
                self.console_text.insert("end", text)
                self.console_text.see("end")
                self.console_text.configure(state="disabled")
            except queue.Empty:
                break
                
        self.after(100, self.poll_log_queue)

    def save_recipe(self):
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if not filename: return
        
        recipe = {
            "yaml_path": self.yaml_entry.get(),
            "project_path": self.project_entry.get(),
            "classes_to_keep": self.get_selected_classes(),
            "weights": self.weights_option.get(),
            "epochs": self.epochs_slider.get(),
            "batch_mode": self.batch_option.get(),
            "batch_custom": self.batch_custom_entry.get(),
            "imgsz": self.imgsz_option.get(),
            "freeze_strategy": self.freeze_strategy_var.get(),
            "stage1_split": self.stage1_split_slider.get(),
            "stage2_lr_scale": self.stage2_lr_scale_entry.get(),
            "custom_freeze": self.custom_freeze_entry.get(),
            "patience_enabled": self.patience_switch.get(),
            "patience": self.patience_entry.get(),
            "optimizer": self.adv_opt_option.get(),
            "lr0": self.adv_lr0_entry.get(),
            "lrf": self.adv_lrf_entry.get(),
            "momentum": self.adv_momentum_entry.get(),
            "weight_decay": self.adv_wd_entry.get(),
            "warmup_epochs": self.adv_warmup_entry.get(),
            "box_loss": self.adv_box_entry.get(),
            "cls_loss": self.adv_cls_entry.get(),
            "dfl_loss": self.adv_dfl_entry.get(),
            "aug_fliplr": self.aug_fliplr_switch.get(),
            "aug_flipud": self.aug_flipud_switch.get(),
            "aug_degrees": self.aug_degrees_slider.get(),
            "aug_scale": self.aug_scale_slider.get(),
            "aug_translate": self.aug_translate_slider.get(),
            "aug_hsv_h": self.aug_hsv_h_slider.get(),
            "aug_hsv_s": self.aug_hsv_s_slider.get(),
            "aug_hsv_v": self.aug_hsv_v_slider.get(),
            "aug_mosaic": self.aug_mosaic_switch.get(),
            "aug_mixup": self.aug_mixup_switch.get(),
            "aug_copypaste": self.aug_copypaste_switch.get(),
            "export_format": self.export_format_option.get(),
            "export_imgsz": self.export_imgsz.get(),
            "export_dynamic": self.dynamic_switch.get(),
            "export_half": self.half_switch.get()
        }
        try:
            with open(filename, 'w', encoding='utf-8') as f: json.dump(recipe, f, indent=4)
            messagebox.showinfo("Success", f"Workflow recipe saved successfully to:\n{os.path.basename(filename)}")
        except Exception as e: messagebox.showerror("Error", f"Failed to save recipe: {str(e)}")

    def load_recipe(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not filename: return
        try:
            with open(filename, 'r', encoding='utf-8') as f: recipe = json.load(f)
            def set_entry(entry_widget, value):
                entry_widget.delete(0, "end")
                entry_widget.insert(0, str(value))
                
            if "yaml_path" in recipe: 
                set_entry(self.yaml_entry, recipe["yaml_path"])
                set_entry(self.ds_yaml_entry, recipe["yaml_path"])
                self.update_class_checkboxes(recipe["yaml_path"])
            if "project_path" in recipe: set_entry(self.project_entry, recipe["project_path"])
            if "classes_to_keep" in recipe:
                kept = recipe["classes_to_keep"]
                if kept is not None:
                    kept_ids = [int(x) for x in kept]
                    for cls_id, var in self.class_vars.items():
                        var.set(1 if cls_id in kept_ids else 0)
                else:
                    self.select_all_classes()
            if "weights" in recipe:
                w_val = recipe["weights"]
                cur_values = self.weights_option.cget("values")
                if w_val not in cur_values:
                    updated_values = [w_val] + [v for v in cur_values if v != "[Custom Weights...]"] + ["[Custom Weights...]"]
                    self.weights_option.configure(values=updated_values)
                self.weights_option.set(w_val)
            if "epochs" in recipe: 
                self.epochs_slider.set(float(recipe["epochs"]))
                self.update_epoch_label(recipe["epochs"])

            if "batch_mode" in recipe:
                self.batch_option.set(recipe["batch_mode"])
                if "batch_custom" in recipe:
                    set_entry(self.batch_custom_entry, recipe["batch_custom"])
                self.on_batch_mode_change(recipe["batch_mode"])
            elif "batch_size" in recipe:
                b_val = str(recipe["batch_size"])
                if b_val in ["4", "8", "16", "32", "64", "-1"]:
                    self.batch_option.set("Auto (-1)" if b_val == "-1" else b_val)
                    self.on_batch_mode_change(self.batch_option.get())
                else:
                    self.batch_option.set("Custom")
                    set_entry(self.batch_custom_entry, b_val)
                    self.on_batch_mode_change("Custom")

            if "freeze_strategy" in recipe:
                self.freeze_strategy_var.set(recipe["freeze_strategy"])
            if "stage1_split" in recipe:
                self.stage1_split_slider.set(float(recipe["stage1_split"]))
                self.stage1_split_lbl.configure(text=f"{int(float(recipe['stage1_split']))}%")
            if "stage2_lr_scale" in recipe:
                set_entry(self.stage2_lr_scale_entry, recipe["stage2_lr_scale"])
            if "custom_freeze" in recipe:
                set_entry(self.custom_freeze_entry, recipe["custom_freeze"])
            self.on_strategy_change()

            if "imgsz" in recipe: self.imgsz_option.set(recipe["imgsz"])
            if "patience_enabled" in recipe:
                if recipe["patience_enabled"] == 1:
                    self.patience_switch.select()
                    self.patience_entry.configure(state="normal")
                else:
                    self.patience_switch.deselect()
                    self.patience_entry.configure(state="disabled")
            if "patience" in recipe: set_entry(self.patience_entry, recipe["patience"])
            
            if "optimizer" in recipe: self.adv_opt_option.set(recipe["optimizer"])
            if "lr0" in recipe: set_entry(self.adv_lr0_entry, recipe["lr0"])
            if "lrf" in recipe: set_entry(self.adv_lrf_entry, recipe["lrf"])
            if "momentum" in recipe: set_entry(self.adv_momentum_entry, recipe["momentum"])
            if "weight_decay" in recipe: set_entry(self.adv_wd_entry, recipe["weight_decay"])
            if "warmup_epochs" in recipe: set_entry(self.adv_warmup_entry, recipe["warmup_epochs"])
            if "box_loss" in recipe: set_entry(self.adv_box_entry, recipe["box_loss"])
            if "cls_loss" in recipe: set_entry(self.adv_cls_entry, recipe["cls_loss"])
            if "dfl_loss" in recipe: set_entry(self.adv_dfl_entry, recipe["dfl_loss"])

            if "aug_fliplr" in recipe:
                self.aug_fliplr_switch.select() if recipe["aug_fliplr"] == 1 else self.aug_fliplr_switch.deselect()
            if "aug_flipud" in recipe:
                self.aug_flipud_switch.select() if recipe["aug_flipud"] == 1 else self.aug_flipud_switch.deselect()
            if "aug_degrees" in recipe:
                self.aug_degrees_slider.set(float(recipe["aug_degrees"]))
                self.aug_degrees_val_lbl.configure(text=f"{int(float(recipe['aug_degrees']))}°")
            if "aug_scale" in recipe:
                self.aug_scale_slider.set(float(recipe["aug_scale"]))
                self.aug_scale_val_lbl.configure(text=f"{float(recipe['aug_scale']):.2f}")
            if "aug_translate" in recipe:
                self.aug_translate_slider.set(float(recipe["aug_translate"]))
                self.aug_translate_val_lbl.configure(text=f"{float(recipe['aug_translate']):.2f}")
            if "aug_hsv_h" in recipe:
                self.aug_hsv_h_slider.set(float(recipe["aug_hsv_h"]))
                self.aug_hsv_h_val_lbl.configure(text=f"{float(recipe['aug_hsv_h']):.3f}")
            if "aug_hsv_s" in recipe:
                self.aug_hsv_s_slider.set(float(recipe["aug_hsv_s"]))
                self.aug_hsv_s_val_lbl.configure(text=f"{float(recipe['aug_hsv_s']):.2f}")
            if "aug_hsv_v" in recipe:
                self.aug_hsv_v_slider.set(float(recipe["aug_hsv_v"]))
                self.aug_hsv_v_val_lbl.configure(text=f"{float(recipe['aug_hsv_v']):.2f}")
            if "aug_mosaic" in recipe:
                self.aug_mosaic_switch.select() if recipe["aug_mosaic"] == 1 else self.aug_mosaic_switch.deselect()
            if "aug_mixup" in recipe:
                self.aug_mixup_switch.select() if recipe["aug_mixup"] == 1 else self.aug_mixup_switch.deselect()
            if "aug_copypaste" in recipe:
                self.aug_copypaste_switch.select() if recipe["aug_copypaste"] == 1 else self.aug_copypaste_switch.deselect()
            
            if "export_format" in recipe: self.export_format_option.set(recipe["export_format"])
            if "export_imgsz" in recipe: self.export_imgsz.set(recipe["export_imgsz"])
            if "export_dynamic" in recipe: self.dynamic_switch.select() if recipe["export_dynamic"] == 1 else self.dynamic_switch.deselect()
            if "export_half" in recipe: self.half_switch.select() if recipe["export_half"] == 1 else self.half_switch.deselect()
                
            messagebox.showinfo("Success", f"Workflow recipe loaded successfully:\n{os.path.basename(filename)}")
        except Exception as e: messagebox.showerror("Error", f"Failed to load recipe: {str(e)}")

    def setup_dataset_manager_tab(self):
        self.tab_dataset.grid_columnconfigure(0, weight=1)
        self.tab_dataset.grid_columnconfigure(1, weight=2)
        self.tab_dataset.grid_rowconfigure(1, weight=1)

        control_frame = ctk.CTkFrame(self.tab_dataset, height=50)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        ctk.CTkLabel(control_frame, text="data.yaml Path:").pack(side="left", padx=10)
        self.ds_yaml_entry = ctk.CTkEntry(control_frame, width=400, placeholder_text="Select data.yaml to analyze dataset...")
        self.ds_yaml_entry.pack(side="left", padx=10, pady=10)
        
        ctk.CTkButton(control_frame, text="Browse YAML", command=self.ds_browse_yaml, fg_color="gray").pack(side="left", padx=5)
        self.ds_analyze_btn = ctk.CTkButton(control_frame, text="Analyze Dataset", command=self.start_dataset_analysis, font=ctk.CTkFont(weight="bold"))
        self.ds_analyze_btn.pack(side="left", padx=20)

        report_frame = ctk.CTkFrame(self.tab_dataset)
        report_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(report_frame, text="Dataset Health Report", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=10, pady=10)
        
        self.ds_report_text = ctk.CTkTextbox(report_frame, font=ctk.CTkFont(family="DejaVu Sans Mono", size=12), wrap="word")
        self.ds_report_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.ds_report_text.insert("0.0", "Awaiting dataset analysis...\n\nSelect a data.yaml file and click 'Analyze Dataset'.")
        self.ds_report_text.configure(state="disabled")

        graph_frame = ctk.CTkFrame(self.tab_dataset)
        graph_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        self.ds_fig = Figure(figsize=(8, 6), dpi=100)
        self.ds_fig.patch.set_facecolor('#2b2b2b')
        self.ds_ax1 = self.ds_fig.add_subplot(211) 
        self.ds_ax2 = self.ds_fig.add_subplot(212) 
        
        for ax in [self.ds_ax1, self.ds_ax2]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values(): spine.set_color('white')

        self.ds_fig.tight_layout(pad=3.0)
        self.ds_canvas = FigureCanvasTkAgg(self.ds_fig, master=graph_frame)
        self.ds_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def ds_browse_yaml(self):
        filename = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
        if filename:
            self.ds_yaml_entry.delete(0, "end")
            self.ds_yaml_entry.insert(0, filename)
            self.yaml_entry.delete(0, "end")
            self.yaml_entry.insert(0, filename)
            self.update_class_checkboxes(filename)

    def start_dataset_analysis(self):
        yaml_path = self.ds_yaml_entry.get()
        if not os.path.exists(yaml_path):
            messagebox.showerror("Error", "Valid data.yaml required for analysis.")
            return
            
        self.ds_analyze_btn.configure(state="disabled", text="Analyzing...")
        self.ds_report_text.configure(state="normal")
        self.ds_report_text.delete("0.0", "end")
        self.ds_report_text.insert("end", "Starting deep analysis of dataset...\nChecking paths, counting labels, and verifying images...\nThis may take a moment.")
        self.ds_report_text.configure(state="disabled")
        
        threading.Thread(target=self.run_dataset_analysis, args=(yaml_path,), daemon=True).start()

    def run_dataset_analysis(self, yaml_path):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            base_dir = os.path.dirname(yaml_path)
            if 'path' in data:
                if os.path.isabs(data['path']): base_dir = data['path']
                else: base_dir = os.path.join(base_dir, data['path'])
                
            train_path = data.get('train', '')
            val_path = data.get('val', '')
            
            if isinstance(train_path, list): train_path = train_path[0]
            if isinstance(val_path, list): val_path = val_path[0]
            
            train_img_dir = os.path.normpath(os.path.join(base_dir, train_path))
            val_img_dir = os.path.normpath(os.path.join(base_dir, val_path))
            train_lbl_dir = train_img_dir.replace('images', 'labels')
            val_lbl_dir = val_img_dir.replace('images', 'labels')
            
            class_names = data.get('names', {})
            if isinstance(class_names, list):
                class_names = {i: name for i, name in enumerate(class_names)}
            num_classes = len(class_names)
            
            stats = {
                'train_imgs': 0, 'val_imgs': 0,
                'missing_lbls': 0, 'corrupt_imgs': 0, 'zero_annots': 0,
                'class_counts': {k: 0 for k in class_names.values()},
                'bboxes_w': [], 'bboxes_h': []
            }

            def scan_split(img_dir, lbl_dir, is_train=True):
                if not os.path.exists(img_dir): return
                valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp')
                images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
                if is_train: stats['train_imgs'] += len(images)
                else: stats['val_imgs'] += len(images)
                
                for img_name in images:
                    img_path = os.path.join(img_dir, img_name)
                    lbl_name = os.path.splitext(img_name)[0] + '.txt'
                    lbl_path = os.path.join(lbl_dir, lbl_name)
                    
                    try:
                        with Image.open(img_path) as im: im.verify()
                    except Exception:
                        stats['corrupt_imgs'] += 1
                        
                    if not os.path.exists(lbl_path):
                        stats['missing_lbls'] += 1
                        continue
                        
                    with open(lbl_path, 'r', encoding='utf-8') as lf:
                        lines = [line.strip() for line in lf.readlines() if line.strip()]
                        if not lines:
                            stats['zero_annots'] += 1
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                w, h = float(parts[3]), float(parts[4])
                                if cls_id in class_names:
                                    c_name = class_names[cls_id]
                                    stats['class_counts'][c_name] += 1
                                    if is_train: 
                                        stats['bboxes_w'].append(w)
                                        stats['bboxes_h'].append(h)

            scan_split(train_img_dir, train_lbl_dir, is_train=True)
            scan_split(val_img_dir, val_lbl_dir, is_train=False)
            
            total_images = stats['train_imgs'] + stats['val_imgs']
            imbalance_detected = False
            counts = list(stats['class_counts'].values())
            if counts and max(counts) > (min(counts) * 4) and min(counts) > 0:
                imbalance_detected = True
                
            report = []
            report.append("DATASET HEALTH")
            report.append("----------------------------")
            report.append(f"Total Images     {total_images:,}")
            report.append(f"Training         {stats['train_imgs']:,}")
            report.append(f"Validation       {stats['val_imgs']:,}")
            report.append(f"Classes          {num_classes}")
            report.append("")
            report.append(f"{'[PASS]' if stats['missing_lbls'] == 0 else '[WARN]'} Missing labels   {stats['missing_lbls']}")
            report.append(f"{'[PASS]' if stats['corrupt_imgs'] == 0 else '[WARN]'} Corrupt images   {stats['corrupt_imgs']}")
            report.append(f"{'[INFO]' if stats['zero_annots'] > 0 else '[PASS]'} Zero annotations {stats['zero_annots']}")
            report.append(f"{'[WARN]' if imbalance_detected else '[PASS]'} Class imbalance  {'Detected' if imbalance_detected else 'Normal'}")
            report.append("\nClass Distribution")
            report.append("----------------------------")
            
            max_c = max(counts) if counts else 1
            for c_name, count in stats['class_counts'].items():
                bars = "#" * int((count / max_c) * 15) if max_c > 0 else ""
                report.append(f"{c_name[:12]:<12} {bars:<15} {count:,}")

            self.after(0, self.update_dataset_ui, report, stats)
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
            self.after(0, lambda: self.ds_analyze_btn.configure(state="normal", text="Analyze Dataset"))

    def update_dataset_ui(self, report_lines, stats):
        self.ds_report_text.configure(state="normal")
        self.ds_report_text.delete("0.0", "end")
        self.ds_report_text.insert("end", "\n".join(report_lines))
        self.ds_report_text.configure(state="disabled")
        
        self.ds_ax1.clear()
        self.ds_ax1.grid(True, linestyle="--", alpha=0.2)
        classes = list(stats['class_counts'].keys())
        counts = list(stats['class_counts'].values())
        self.ds_ax1.bar(classes, counts, color="#3b82f6")
        self.ds_ax1.set_title("Class Object Counts", color='white')
        self.ds_ax1.tick_params(axis='x', rotation=45, colors='white')
        
        self.ds_ax2.clear()
        self.ds_ax2.grid(True, linestyle="--", alpha=0.2)
        if stats['bboxes_w']:
            w = stats['bboxes_w'][:5000]
            h = stats['bboxes_h'][:5000]
            self.ds_ax2.scatter(w, h, alpha=0.3, color="#2ef072", s=5)
            self.ds_ax2.set_xlim(0, 1.0)
            self.ds_ax2.set_ylim(0, 1.0)
        self.ds_ax2.set_title("Bounding Box Size Dist (Width vs Height)", color='white')
        self.ds_ax2.set_xlabel("Width (Normalized)", color='white')
        self.ds_ax2.set_ylabel("Height (Normalized)", color='white')
        self.ds_ax2.tick_params(colors='white')
        
        self.ds_fig.tight_layout(pad=2.0)
        self.ds_canvas.draw()
        
        self.ds_analyze_btn.configure(state="normal", text="Analyze Dataset")

    def setup_classes_tab(self):
        self.tab_classes.grid_columnconfigure(0, weight=1)
        
        classes_frame = ctk.CTkFrame(self.tab_classes)
        classes_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(classes_frame, text="Class Filtering (Optional)", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))
        
        info_text = (
            "Tick the classes you want to train on. Unticked classes will be ignored during training.\n"
            "This is extremely useful if you want to exclude certain items without modifying your dataset's text files.\n\n"
            "(Load a data.yaml file to populate this list automatically.)"
        )
        ctk.CTkLabel(classes_frame, text=info_text, justify="left").pack(anchor="w", padx=20, pady=(0, 15))
        
        btn_frame = ctk.CTkFrame(classes_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(btn_frame, text="Select All", width=100, command=self.select_all_classes, fg_color="#3b82f6").pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Deselect All", width=100, command=self.deselect_all_classes, fg_color="#ef4444").pack(side="left")

        self.classes_scroll_frame = ctk.CTkScrollableFrame(classes_frame)
        self.classes_scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        ctk.CTkLabel(self.classes_scroll_frame, text="No data.yaml loaded yet.", text_color="gray").pack(pady=20)

    def select_all_classes(self):
        for var in self.class_vars.values():
            var.set(1)

    def deselect_all_classes(self):
        for var in self.class_vars.values():
            var.set(0)

    def update_class_checkboxes(self, yaml_path):
        if not yaml_path or not os.path.exists(yaml_path):
            return
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            names = data.get('names', {})
            if isinstance(names, list):
                names = {i: name for i, name in enumerate(names)}
            
            self.class_names = names
            
            for widget in self.classes_scroll_frame.winfo_children():
                widget.destroy()
                
            self.class_vars = {}
            if not names:
                ctk.CTkLabel(self.classes_scroll_frame, text="No classes found in yaml.", text_color="red").pack(pady=10)
                return

            for cls_id, cls_name in names.items():
                var = ctk.IntVar(value=1)
                self.class_vars[cls_id] = var
                cb = ctk.CTkCheckBox(self.classes_scroll_frame, text=f"[{cls_id}] {cls_name}", variable=var, font=ctk.CTkFont(size=14))
                cb.pack(anchor="w", pady=6, padx=10)
        except Exception as e:
            print(f"Error parsing classes: {e}")

    def get_selected_classes(self):
        if not self.class_vars:
            return None
        selected = [cls_id for cls_id, var in self.class_vars.items() if var.get() == 1]
        if len(selected) == len(self.class_vars):
            return None
        return selected

    def setup_training_tab(self):
        self.tab_train.grid_columnconfigure(0, weight=1)
        self.tab_train.grid_columnconfigure(1, weight=2)

        config_frame = ctk.CTkFrame(self.tab_train)
        config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(config_frame, text="Paths & Configuration", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))

        self.yaml_entry = ctk.CTkEntry(config_frame, placeholder_text="Path to data.yaml")
        self.yaml_entry.pack(fill="x", padx=15, pady=5)
        ctk.CTkButton(config_frame, text="Browse YAML", command=self.browse_yaml, fg_color="gray").pack(anchor="e", padx=15, pady=2)

        self.project_entry = ctk.CTkEntry(config_frame, placeholder_text="Output Folder (Default: runs/detect)")
        self.project_entry.pack(fill="x", padx=15, pady=5)
        ctk.CTkButton(config_frame, text="Browse Folder", command=self.browse_output_dir, fg_color="gray").pack(anchor="e", padx=15, pady=2)

        ctk.CTkLabel(config_frame, text="Starting Weights", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=15, pady=(10, 2))
        weights_row = ctk.CTkFrame(config_frame, fg_color="transparent")
        weights_row.pack(fill="x", padx=15, pady=5)

        self.weights_option = ctk.CTkOptionMenu(
            weights_row, 
            values=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "[Custom Weights...]"],
            command=self.on_weights_select
        )
        self.weights_option.pack(side="left", fill="x", expand=True, padx=(0, 6))

        browse_pt_btn = ctk.CTkButton(
            weights_row, text="Browse", width=80, fg_color="#475569", hover_color="#334155",
            command=self.browse_starting_weights
        )
        browse_pt_btn.pack(side="right")
        CTkToolTip(browse_pt_btn, "Load custom pretrained weights (.pt) from your disk or previous training runs.")

        ctk.CTkLabel(config_frame, text="Base Hyperparameters", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))

        epochs_header_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        epochs_header_frame.pack(fill="x", padx=15, pady=(5, 0))

        ctk.CTkLabel(epochs_header_frame, text="Epochs:").pack(side="left")
        self.epoch_value_label = ctk.CTkLabel(
            epochs_header_frame, text="100 Epochs", font=ctk.CTkFont(size=12, weight="bold"), text_color="#2ef072"
        )
        self.epoch_value_label.pack(side="right")

        self.epochs_slider = ctk.CTkSlider(
            config_frame, from_=5, to=300, number_of_steps=59, command=self.update_epoch_label
        )
        self.epochs_slider.set(100)
        self.epochs_slider.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(config_frame, text="Batch Size:").pack(anchor="w", padx=15)
        batch_input_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        batch_input_frame.pack(fill="x", padx=15, pady=5)

        self.batch_option = ctk.CTkOptionMenu(
            batch_input_frame,
            values=["Auto (-1)", "4", "8", "16", "32", "64", "Custom"],
            command=self.on_batch_mode_change,
            width=135
        )
        self.batch_option.set("32")
        self.batch_option.pack(side="left")

        self.batch_custom_entry = ctk.CTkEntry(
            batch_input_frame,
            placeholder_text="e.g. 12, 24",
            width=95
        )
        self.batch_custom_entry.pack(side="left", padx=(10, 0))
        self.batch_custom_entry.configure(state="disabled")
        CTkToolTip(self.batch_custom_entry, "Specify any custom batch size (e.g. 1, 2, 12, 24, 48) or -1 for Auto-batch.")

        ctk.CTkLabel(config_frame, text="Image Size (imgsz):").pack(anchor="w", padx=15)
        self.imgsz_option = ctk.CTkSegmentedButton(config_frame, values=["416", "640", "1024"])
        self.imgsz_option.set("640")
        self.imgsz_option.pack(fill="x", padx=15, pady=5)

        patience_card = ctk.CTkFrame(config_frame, fg_color="transparent")
        patience_card.pack(fill="x", padx=15, pady=(15, 5))
        patience_card.grid_columnconfigure(0, weight=1)
        patience_card.grid_columnconfigure(1, weight=1)

        self.patience_switch = ctk.CTkSwitch(
            patience_card, text="Early Stopping", command=self.toggle_patience_entry
        )
        self.patience_switch.select()
        self.patience_switch.grid(row=0, column=0, sticky="w", pady=(5, 0))

        self.patience_entry = ctk.CTkEntry(patience_card, placeholder_text="50")
        self.patience_entry.insert(0, "50")
        self.patience_entry.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=20)
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)

        self.train_btn = ctk.CTkButton(
            btn_frame, text="Launch", font=ctk.CTkFont(size=14, weight="bold"), command=self.start_training_thread
        )
        self.train_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.pause_btn = ctk.CTkButton(
            btn_frame, text="|| Pause", font=ctk.CTkFont(size=14, weight="bold"), 
            command=self.toggle_pause, state="disabled", fg_color="#eab308", hover_color="#ca8a04"
        )
        self.pause_btn.grid(row=0, column=1, padx=5, sticky="ew")

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="[ ] Stop", font=ctk.CTkFont(size=14, weight="bold"), 
            command=self.request_stop, state="disabled", fg_color="#ef4444", hover_color="#b91c1c"
        )
        self.stop_btn.grid(row=0, column=2, padx=(5, 0), sticky="ew")

        console_frame = ctk.CTkFrame(self.tab_train)
        console_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        stats_header_frame = ctk.CTkFrame(console_frame, fg_color="transparent")
        stats_header_frame.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(stats_header_frame, text="Live Progress & Stats", font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        
        self.benchmark_btn = ctk.CTkButton(
            stats_header_frame, 
            text="Benchmark", 
            font=ctk.CTkFont(size=11, weight="bold"),
            width=95, 
            height=26,
            fg_color="#3b82f6", 
            hover_color="#1d4ed8",
            command=self.run_hardware_benchmark
        )
        self.benchmark_btn.pack(side="right")

        progress_card = ctk.CTkFrame(console_frame)
        progress_card.pack(fill="x", padx=15, pady=5)

        self.progress_bar = ctk.CTkProgressBar(progress_card, height=18)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", padx=15, pady=(12, 8))

        self.progress_text = ctk.CTkLabel(progress_card, text="Epoch: 0 / 0 (0%)", font=ctk.CTkFont(size=12, weight="bold"))
        self.progress_text.pack(anchor="w", padx=15, pady=2)

        timer_subframe = ctk.CTkFrame(progress_card, fg_color="transparent")
        timer_subframe.pack(fill="x", padx=15, pady=(2, 10))

        self.elapsed_label = ctk.CTkLabel(timer_subframe, text="Elapsed: 00:00:00", font=ctk.CTkFont(size=11), text_color="#3b82f6")
        self.elapsed_label.pack(side="left")

        self.eta_label = ctk.CTkLabel(timer_subframe, text="ETA: --:--:--", font=ctk.CTkFont(size=11), text_color="#2ef072")
        self.eta_label.pack(side="right")

        ctk.CTkLabel(console_frame, text="Live Output Log", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=15, pady=(10, 2))
        self.console_text = ctk.CTkTextbox(console_frame, wrap="none", font=ctk.CTkFont(family="DejaVu Sans Mono", size=11))
        self.console_text.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.console_text.configure(state="disabled")

    def create_param_header(self, parent, label_text, tooltip_desc, rec_val="", row=None, col=None):
        header_subframe = ctk.CTkFrame(parent, fg_color="transparent")
        if row is not None and col is not None:
            header_subframe.grid(row=row, column=col, sticky="w", padx=15, pady=(5, 2))
        else:
            header_subframe.pack(fill="x", padx=15, pady=(8, 2))

        ctk.CTkLabel(
            header_subframe, 
            text=label_text, 
            font=ctk.CTkFont(size=12)
        ).pack(side="left")

        full_tip = f"{tooltip_desc}\n\nRecommended: {rec_val}" if rec_val else tooltip_desc

        tip_holder = [None]
        def on_info_click():
            if tip_holder[0]:
                tip_holder[0].hide_tooltip()
            messagebox.showinfo(label_text.replace(":", ""), full_tip)

        info_btn = ctk.CTkButton(
            header_subframe,
            text="i",
            width=18,
            height=18,
            corner_radius=9,
            fg_color="#334155",
            hover_color="#475569",
            text_color="#38bdf8",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=on_info_click
        )
        info_btn.pack(side="left", padx=6)
        tip_holder[0] = CTkToolTip(info_btn, full_tip)
        return header_subframe

    def setup_advanced_hyperparams_tab(self):
        self.tab_adv.grid_columnconfigure(0, weight=1)
        self.tab_adv.grid_columnconfigure(1, weight=1)

        opt_lr_frame = ctk.CTkFrame(self.tab_adv)
        opt_lr_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(opt_lr_frame, text="Optimization Engine", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))
        
        self.create_param_header(
            opt_lr_frame,
            "Optimizer:",
            "Choice of gradient descent optimizer. 'auto' selects the best algorithm for the model scale.",
            "auto (or AdamW for small datasets/fine-tuning, SGD for scratch training)"
        )
        self.adv_opt_option = ctk.CTkOptionMenu(opt_lr_frame, values=["auto", "AdamW", "MuSGD", "SGD", "Adam", "RMSProp"])
        self.adv_opt_option.set("auto")
        self.adv_opt_option.pack(fill="x", padx=15, pady=5)
        
        self.create_param_header(
            opt_lr_frame,
            "Initial LR (lr0):",
            "Controls the initial learning rate. Higher values can train faster but may cause unstable divergence.",
            "0.01 for SGD, 0.001 - 0.002 for AdamW"
        )
        self.adv_lr0_entry = ctk.CTkEntry(opt_lr_frame, placeholder_text="0.01")
        self.adv_lr0_entry.insert(0, "0.01")
        self.adv_lr0_entry.pack(fill="x", padx=15, pady=5)
        
        self.create_param_header(
            opt_lr_frame,
            "Final LR Fraction (lrf):",
            "Final learning rate ratio relative to lr0 via cosine scheduling (final_lr = lr0 * lrf).",
            "0.01 (decays to 1% of initial rate)"
        )
        self.adv_lrf_entry = ctk.CTkEntry(opt_lr_frame, placeholder_text="0.01")
        self.adv_lrf_entry.insert(0, "0.01")
        self.adv_lrf_entry.pack(fill="x", padx=15, pady=5)

        mech_frame = ctk.CTkFrame(self.tab_adv)
        mech_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(mech_frame, text="Momentum & Warmup", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))

        self.create_param_header(
            mech_frame,
            "Momentum:",
            "Momentum factor for SGD or beta1 for AdamW. Accelerates gradient vectors in the right directions.",
            "0.937"
        )
        self.adv_momentum_entry = ctk.CTkEntry(mech_frame, placeholder_text="0.937")
        self.adv_momentum_entry.insert(0, "0.937")
        self.adv_momentum_entry.pack(fill="x", padx=15, pady=5)
        
        self.create_param_header(
            mech_frame,
            "Weight Decay:",
            "L2 weight regularization penalty that prevents weights from becoming too large and overfitting.",
            "0.0005"
        )
        self.adv_wd_entry = ctk.CTkEntry(mech_frame, placeholder_text="0.0005")
        self.adv_wd_entry.insert(0, "0.0005")
        self.adv_wd_entry.pack(fill="x", padx=15, pady=5)
        
        self.create_param_header(
            mech_frame,
            "Warmup Epochs:",
            "Epochs spent gradually scaling up the learning rate from 0 to prevent gradient explosions early on.",
            "3.0 epochs"
        )
        self.adv_warmup_entry = ctk.CTkEntry(mech_frame, placeholder_text="3.0")
        self.adv_warmup_entry.insert(0, "3.0")
        self.adv_warmup_entry.pack(fill="x", padx=15, pady=5)

        strategy_frame = ctk.CTkFrame(mech_frame, fg_color="#1e1e1e", corner_radius=8)
        strategy_frame.pack(fill="x", padx=15, pady=(10, 5))

        self.create_param_header(
            strategy_frame,
            "Training Strategy:",
            "Controls network layer locking and automated multi-stage transfer learning.\n\n"
            "• Full Fine-Tuning: Trains all parameters end-to-end.\n"
            "• Freeze Backbone: Keeps foundational visual features intact (trains neck + head).\n"
            "• Freeze Backbone + Neck: Trains ONLY the detection head (True Linear Probing).\n"
            "• Two-Stage Transfer Learning: Automatically trains Stage 1 with frozen backbone, "
            "then unfreezes all layers with reduced LR for Stage 2.\n"
            "• Custom: Specify exact number of layers to freeze."
        )

        self.freeze_strategy_var = ctk.StringVar(value="Full Fine-Tuning")

        def on_strategy_change():
            strat = self.freeze_strategy_var.get()
            if strat == "Custom":
                self.custom_freeze_entry.configure(state="normal")
            else:
                self.custom_freeze_entry.configure(state="disabled")
            
            if strat == "Two-Stage Transfer Learning":
                self.two_stage_config_frame.configure(fg_color="#111827")
                self.stage1_split_slider.configure(state="normal")
                self.stage2_lr_scale_entry.configure(state="normal")
            else:
                self.two_stage_config_frame.configure(fg_color="#18181b")
                self.stage1_split_slider.configure(state="disabled")
                self.stage2_lr_scale_entry.configure(state="disabled")

        self.on_strategy_change = on_strategy_change

        ctk.CTkRadioButton(
            strategy_frame,
            text="Full Fine-Tuning (Single Stage)",
            value="Full Fine-Tuning",
            variable=self.freeze_strategy_var,
            command=on_strategy_change
        ).pack(anchor="w", padx=15, pady=3)

        ctk.CTkRadioButton(
            strategy_frame,
            text="Freeze Backbone (Layers 0-9)",
            value="Freeze Backbone",
            variable=self.freeze_strategy_var,
            command=on_strategy_change
        ).pack(anchor="w", padx=15, pady=3)

        ctk.CTkRadioButton(
            strategy_frame,
            text="Freeze Backbone + Neck (Head Only / Linear Probing)",
            value="Freeze Backbone + Neck",
            variable=self.freeze_strategy_var,
            command=on_strategy_change
        ).pack(anchor="w", padx=15, pady=3)

        ctk.CTkRadioButton(
            strategy_frame,
            text="Two-Stage Transfer Learning (Automated: Frozen -> Full)",
            value="Two-Stage Transfer Learning",
            variable=self.freeze_strategy_var,
            command=on_strategy_change
        ).pack(anchor="w", padx=15, pady=3)

        self.two_stage_config_frame = ctk.CTkFrame(strategy_frame, fg_color="#18181b", corner_radius=6)
        self.two_stage_config_frame.pack(fill="x", padx=(30, 15), pady=(4, 8))

        budget_row = ctk.CTkFrame(self.two_stage_config_frame, fg_color="transparent")
        budget_row.pack(fill="x", padx=10, pady=(6, 2))
        ctk.CTkLabel(budget_row, text="Stage 1 Epoch Budget:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.stage1_split_lbl = ctk.CTkLabel(budget_row, text="30%", font=ctk.CTkFont(size=11, weight="bold"), text_color="#38bdf8")
        self.stage1_split_lbl.pack(side="right")

        self.stage1_split_slider = ctk.CTkSlider(
            self.two_stage_config_frame, from_=10, to=70, number_of_steps=12,
            command=lambda v: self.stage1_split_lbl.configure(text=f"{int(v)}%")
        )
        self.stage1_split_slider.set(30)
        self.stage1_split_slider.pack(fill="x", padx=10, pady=(0, 6))

        lr_scale_row = ctk.CTkFrame(self.two_stage_config_frame, fg_color="transparent")
        lr_scale_row.pack(fill="x", padx=10, pady=(2, 8))
        ctk.CTkLabel(lr_scale_row, text="Stage 2 LR Multiplier:", font=ctk.CTkFont(size=11)).pack(side="left")
        self.stage2_lr_scale_entry = ctk.CTkEntry(lr_scale_row, width=65)
        self.stage2_lr_scale_entry.insert(0, "0.10")
        self.stage2_lr_scale_entry.pack(side="right")
        CTkToolTip(self.stage2_lr_scale_entry, "Multiplier applied to base lr0 during Stage 2 full fine-tuning (e.g. 0.10 = 10% of Stage 1 lr0).")

        ctk.CTkRadioButton(
            strategy_frame,
            text="Custom Freeze Layers",
            value="Custom",
            variable=self.freeze_strategy_var,
            command=on_strategy_change
        ).pack(anchor="w", padx=15, pady=3)

        custom_box = ctk.CTkFrame(strategy_frame, fg_color="transparent")
        custom_box.pack(fill="x", padx=(30, 15), pady=(2, 10))

        ctk.CTkLabel(custom_box, text="Custom Freeze Depth (N):", font=ctk.CTkFont(size=11)).pack(side="left")
        self.custom_freeze_entry = ctk.CTkEntry(custom_box, width=75, placeholder_text="e.g. 15")
        self.custom_freeze_entry.insert(0, "0")
        self.custom_freeze_entry.configure(state="disabled")
        self.custom_freeze_entry.pack(side="left", padx=10)
        CTkToolTip(self.custom_freeze_entry, "Freezes layers 0 up to N-1. Typical YOLO architectures: Backbone is 0-9, Neck is 10-21.")

        on_strategy_change()

        loss_frame = ctk.CTkFrame(self.tab_adv)
        loss_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        loss_frame.grid_columnconfigure(0, weight=1)
        loss_frame.grid_columnconfigure(1, weight=1)
        loss_frame.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(loss_frame, text="Loss Component Gains", font=ctk.CTkFont(size=15, weight="bold")).grid(row=0, column=0, columnspan=3, sticky="w", padx=15, pady=(15, 5))

        self.create_param_header(
            loss_frame,
            "Box Loss (box):",
            "Weight gain for bounding box regression (GIoU/CIoU loss). Higher values prioritize box alignment precision.",
            "7.5",
            row=1, col=0
        )
        self.adv_box_entry = ctk.CTkEntry(loss_frame, placeholder_text="7.5")
        self.adv_box_entry.insert(0, "7.5")
        self.adv_box_entry.grid(row=2, column=0, sticky="ew", padx=15, pady=5)

        self.create_param_header(
            loss_frame,
            "Cls Loss (cls):",
            "Weight gain for classification cross-entropy loss. Increase if model misclassifies object categories.",
            "0.5",
            row=1, col=1
        )
        self.adv_cls_entry = ctk.CTkEntry(loss_frame, placeholder_text="0.5")
        self.adv_cls_entry.insert(0, "0.5")
        self.adv_cls_entry.grid(row=2, column=1, sticky="ew", padx=15, pady=5)

        self.create_param_header(
            loss_frame,
            "DFL Loss (dfl):",
            "Weight gain for Distribution Focal Loss used in fine-grained bounding box boundary estimation.",
            "1.5",
            row=1, col=2
        )
        self.adv_dfl_entry = ctk.CTkEntry(loss_frame, placeholder_text="1.5")
        self.adv_dfl_entry.insert(0, "1.5")
        self.adv_dfl_entry.grid(row=2, column=2, sticky="ew", padx=15, pady=5)

    def setup_augmentation_tab(self):
        self.tab_aug.grid_columnconfigure(0, weight=1)
        self.tab_aug.grid_columnconfigure(1, weight=1)
        self.tab_aug.grid_rowconfigure((0, 1), weight=1)

        def create_slider_row(parent, label_text, tip_desc, rec_val, from_val, to_val, default_val, format_fn):
            self.create_param_header(parent, label_text, tip_desc, rec_val)
            sub = ctk.CTkFrame(parent, fg_color="transparent")
            sub.pack(fill="x", padx=15, pady=(2, 8))
            val_lbl = ctk.CTkLabel(sub, text=format_fn(default_val), font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8", width=55, anchor="e")
            slider = ctk.CTkSlider(sub, from_=from_val, to=to_val, number_of_steps=100)
            slider.set(default_val)
            slider.configure(command=lambda v: val_lbl.configure(text=format_fn(v)))
            slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
            val_lbl.pack(side="right")
            return slider, val_lbl

        geo_frame = ctk.CTkFrame(self.tab_aug)
        geo_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(geo_frame, text="Geometric & Spatial Transformations", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 8))

        flips_sub = ctk.CTkFrame(geo_frame, fg_color="#1e1e1e", corner_radius=8)
        flips_sub.pack(fill="x", padx=15, pady=(5, 10))
        flips_sub.grid_columnconfigure((0, 1), weight=1)

        self.aug_fliplr_switch = ctk.CTkSwitch(flips_sub, text="Horizontal Flip")
        self.aug_fliplr_switch.select()
        self.aug_fliplr_switch.grid(row=0, column=0, padx=15, pady=12, sticky="w")
        CTkToolTip(self.aug_fliplr_switch, "Probability of flipping image horizontally (p=0.5). Great for general orientation invariance.")

        self.aug_flipud_switch = ctk.CTkSwitch(flips_sub, text="Vertical Flip")
        self.aug_flipud_switch.deselect()
        self.aug_flipud_switch.grid(row=0, column=1, padx=15, pady=12, sticky="w")
        CTkToolTip(self.aug_flipud_switch, "Probability of flipping image vertically (p=0.5). Recommended for top-down camera views, inspectable surfaces.")

        self.aug_degrees_slider, self.aug_degrees_val_lbl = create_slider_row(
            geo_frame, "Rotation (degrees):", "Random rotation range (+/- degrees).", "10.0° (or 0.0° if orientation is fixed)", 0.0, 180.0, 10.0, lambda v: f"{int(v)}°"
        )
        self.aug_scale_slider, self.aug_scale_val_lbl = create_slider_row(
            geo_frame, "Scale (scale):", "Random image scaling factor gain (+/- fraction).", "0.5", 0.0, 1.0, 0.5, lambda v: f"{v:.2f}"
        )
        self.aug_translate_slider, self.aug_translate_val_lbl = create_slider_row(
            geo_frame, "Translation (translate):", "Random image translation offset (+/- fraction of image dimensions).", "0.1", 0.0, 0.5, 0.1, lambda v: f"{v:.2f}"
        )

        hsv_frame = ctk.CTkFrame(self.tab_aug)
        hsv_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(hsv_frame, text="Color Space Invariance (HSV)", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 8))

        self.aug_hsv_h_slider, self.aug_hsv_h_val_lbl = create_slider_row(
            hsv_frame, "HSV-Hue (hsv_h):", "Random hue color shift (+/- fraction of color wheel). Keep low to avoid altering tint severely.", "0.015", 0.0, 0.1, 0.015, lambda v: f"{v:.3f}"
        )
        self.aug_hsv_s_slider, self.aug_hsv_s_val_lbl = create_slider_row(
            hsv_frame, "HSV-Saturation (hsv_s):", "Random saturation shift (+/- fraction). Handles variable camera sensor color warmth.", "0.7", 0.0, 1.0, 0.7, lambda v: f"{v:.2f}"
        )
        self.aug_hsv_v_slider, self.aug_hsv_v_val_lbl = create_slider_row(
            hsv_frame, "HSV-Value/Brightness (hsv_v):", "Random brightness shift (+/- fraction). Improves robustness against shadows and lighting changes.", "0.4", 0.0, 1.0, 0.4, lambda v: f"{v:.2f}"
        )

        comp_frame = ctk.CTkFrame(self.tab_aug)
        comp_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(comp_frame, text="Multi-Image Mosaic & Composition Engine", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=(15, 8))

        comp_grid = ctk.CTkFrame(comp_frame, fg_color="#1e1e1e", corner_radius=8)
        comp_grid.pack(fill="x", padx=15, pady=(5, 15))
        comp_grid.grid_columnconfigure((0, 1, 2), weight=1)

        def add_comp_toggle(parent, col, title, desc, default_on=False):
            sub = ctk.CTkFrame(parent, fg_color="transparent")
            sub.grid(row=0, column=col, padx=15, pady=12, sticky="w")
            sw = ctk.CTkSwitch(sub, text=title)
            if default_on: sw.select()
            else: sw.deselect()
            sw.pack(side="left")
            CTkToolTip(sw, desc)
            return sw

        self.aug_mosaic_switch = add_comp_toggle(
            comp_grid, 0, "Mosaic (4-image mix)", "Combines 4 training images into one mosaic. Prevents overfitting and teaches small object detection.", default_on=True
        )
        self.aug_mixup_switch = add_comp_toggle(
            comp_grid, 1, "MixUp", "Blends two different images with transparency. Encourages softer decision boundaries.", default_on=False
        )
        self.aug_copypaste_switch = add_comp_toggle(
            comp_grid, 2, "Copy-Paste", "Copies object instances and pastes them onto other background images. Great for rare defects.", default_on=False
        )

    def toggle_patience_entry(self):
        if self.patience_switch.get() == 1:
            self.patience_entry.configure(state="normal")
        else:
            self.patience_entry.configure(state="disabled")

    def on_batch_mode_change(self, mode):
        if mode == "Custom":
            self.batch_custom_entry.configure(state="normal")
            self.batch_custom_entry.focus()
        else:
            self.batch_custom_entry.configure(state="disabled")

    def get_batch_size(self):
        mode = self.batch_option.get()
        if mode == "Auto (-1)":
            return -1
        elif mode == "Custom":
            raw = self.batch_custom_entry.get().strip()
            if not raw:
                raise ValueError("Custom batch size cannot be empty. Specify an integer or -1 for Auto.")
            val = int(raw)
            if val <= 0 and val != -1:
                raise ValueError(f"Invalid custom batch size '{val}'. Must be a positive integer or -1 for Auto.")
            return val
        else:
            return int(mode)

    def on_weights_select(self, choice):
        if choice == "[Custom Weights...]":
            self.browse_starting_weights()

    def browse_starting_weights(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt")])
        if filename:
            cur_values = self.weights_option.cget("values")
            if filename not in cur_values:
                updated_values = [filename] + [v for v in cur_values if v != "[Custom Weights...]"] + ["[Custom Weights...]"]
                self.weights_option.configure(values=updated_values)
            self.weights_option.set(filename)
        else:
            if self.weights_option.get() == "[Custom Weights...]":
                self.weights_option.set("yolo26n.pt")

    def update_epoch_label(self, value):
        self.epoch_value_label.configure(text=f"{int(value)} Epochs")

    def run_hardware_benchmark(self):
        self.benchmark_btn.configure(state="disabled", text="Testing...")
        
        def benchmark_task():
            gpu_available = torch.cuda.is_available()
            vram_gb = 0
            
            if gpu_available:
                try:
                    gpu_properties = torch.cuda.get_device_properties(0)
                    vram_gb = gpu_properties.total_memory / (1024 ** 3)  
                except Exception:
                    vram_gb = 4.0  
                
                if vram_gb >= 12.0:
                    rec_batch, rec_imgsz = "64", "640"
                    mode_str = f"High-End GPU Detected ({vram_gb:.1f} GB VRAM)"
                elif vram_gb >= 8.0:
                    rec_batch, rec_imgsz = "32", "640"
                    mode_str = f"Mid-Range GPU Detected ({vram_gb:.1f} GB VRAM)"
                elif vram_gb >= 4.0:
                    rec_batch, rec_imgsz = "16", "640"
                    mode_str = f"Entry-Level GPU Detected ({vram_gb:.1f} GB VRAM)"
                else:
                    rec_batch, rec_imgsz = "8", "416"
                    mode_str = f"Low VRAM GPU Detected ({vram_gb:.1f} GB VRAM)"
            else:
                rec_batch, rec_imgsz = "8", "416"
                mode_str = "CPU Mode Active (No GPU Detected)"

            def apply_results():
                self.batch_option.set(rec_batch)
                self.on_batch_mode_change(rec_batch)
                self.imgsz_option.set(rec_imgsz)
                self.benchmark_btn.configure(state="normal", text="Benchmark")
                
                messagebox.showinfo(
                    "Benchmark Complete", 
                    f"Hardware Benchmark Analysis:\n\n"
                    f"• Device Mode: {mode_str}\n"
                    f"• Recommended Batch Size: {rec_batch}\n"
                    f"• Recommended Image Size: {rec_imgsz} px\n\n"
                    f"The optimal settings have been automatically applied to your configuration panel to prevent OOM errors."
                )

            self.after(0, apply_results)

        threading.Thread(target=benchmark_task, daemon=True).start()

    def setup_metrics_tab(self):
        self.tab_metrics.grid_columnconfigure(0, weight=3)
        self.tab_metrics.grid_columnconfigure(1, weight=1, minsize=320)
        self.tab_metrics.grid_rowconfigure(1, weight=1)

        control_bar = ctk.CTkFrame(self.tab_metrics, height=45)
        control_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        ctk.CTkButton(control_bar, text="Load results.csv", command=self.load_results_csv).pack(side="left", padx=10, pady=8)
        ctk.CTkButton(control_bar, text="View Advanced Curves", command=self.open_analytics_popup, fg_color="#8b5cf6", hover_color="#7c3aed").pack(side="left", padx=10, pady=8)
        ctk.CTkButton(control_bar, text="Export PDF Report", command=self.export_pdf_report, fg_color="#10b981", hover_color="#059669").pack(side="left", padx=10, pady=8)

        self.metrics_status = ctk.CTkLabel(control_bar, text="No metrics loaded yet.", font=ctk.CTkFont(size=13))
        self.metrics_status.pack(side="right", padx=15)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.fig.patch.set_facecolor('#2b2b2b')
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.set_title("Training Performance (mAP Metrics)")

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.tab_metrics)
        self.chart_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)

        self.metrics_panel = ctk.CTkFrame(self.tab_metrics, corner_radius=10)
        self.metrics_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.metrics_panel.grid_columnconfigure((0, 1), weight=1)

        header_card = ctk.CTkFrame(self.metrics_panel, fg_color="transparent")
        header_card.pack(fill="x", padx=12, pady=(12, 8))
        
        ctk.CTkLabel(
            header_card, 
            text="Live Telemetry", 
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left")

        self.card_status_badge = ctk.CTkLabel(
            header_card, 
            text="STANDBY", 
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#94a3b8",
            fg_color="#334155",
            corner_radius=6,
            padx=8,
            pady=2
        )
        self.card_status_badge.pack(side="right")

        cards_grid = ctk.CTkFrame(self.metrics_panel, fg_color="transparent")
        cards_grid.pack(fill="x", padx=10, pady=5)
        cards_grid.grid_columnconfigure((0, 1), weight=1)

        def create_mini_card(parent, row, col, title, initial_val, val_color="#ffffff"):
            box = ctk.CTkFrame(parent, corner_radius=8, fg_color="#1e1e1e")
            box.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            ctk.CTkLabel(box, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color="#94a3b8").pack(anchor="w", padx=10, pady=(6, 1))
            val_lbl = ctk.CTkLabel(box, text=initial_val, font=ctk.CTkFont(size=17, weight="bold"), text_color=val_color)
            val_lbl.pack(anchor="w", padx=10, pady=(0, 6))
            return val_lbl

        self.card_epoch_val = create_mini_card(cards_grid, 0, 0, "Epoch", "-- / --", "#ffffff")
        self.card_map50_val = create_mini_card(cards_grid, 0, 1, "mAP@50", "--", "#2ef072")
        self.card_map50_95_val = create_mini_card(cards_grid, 1, 0, "mAP@50-95", "--", "#3b82f6")
        self.card_loss_val = create_mini_card(cards_grid, 1, 1, "Loss", "--", "#f59e0b")

        detail_card = ctk.CTkFrame(self.metrics_panel, corner_radius=8, fg_color="#1e1e1e")
        detail_card.pack(fill="both", expand=True, padx=14, pady=(8, 12))

        def add_stat_row(parent, label_text):
            row_frame = ctk.CTkFrame(parent, fg_color="transparent")
            row_frame.pack(fill="x", padx=12, pady=7)
            ctk.CTkLabel(row_frame, text=label_text, font=ctk.CTkFont(size=12), text_color="#cbd5e1").pack(side="left")
            val_label = ctk.CTkLabel(row_frame, text="--", font=ctk.CTkFont(size=12, weight="bold"), text_color="#ffffff")
            val_label.pack(side="right")
            return val_label

        self.card_best_map50_val = add_stat_row(detail_card, "Best mAP@50")
        self.card_best_epoch_val = add_stat_row(detail_card, "Best Epoch")
        self.card_lr_val = add_stat_row(detail_card, "Current LR")
        self.card_gpu_util_val = add_stat_row(detail_card, "GPU Utilization")
        self.card_vram_val = add_stat_row(detail_card, "VRAM Usage")

    def get_gpu_telemetry(self):
        if not torch.cuda.is_available():
            return "N/A (CPU)", "0.0 / 0.0 GB"

        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            reserved_mem = torch.cuda.memory_reserved(0) / (1024 ** 3)
            vram_str = f"{reserved_mem:.1f} / {total_mem:.1f} GB"

            gpu_util_str = "Active"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util_str = f"{rates.gpu}%"
            except Exception:
                pct = int((reserved_mem / total_mem) * 100) if total_mem > 0 else 0
                gpu_util_str = f"{pct}% (VRAM Load)"

            return gpu_util_str, vram_str
        except Exception:
            return "Active", "-- / -- GB"

    def update_live_metrics_cards(self, df):
        try:
            if df.empty:
                return

            last_row = df.iloc[-1]
            total_epochs = self.total_epochs

            current_epoch = int(last_row['epoch']) if 'epoch' in df.columns else len(df)
            self.card_epoch_val.configure(text=f"{current_epoch} / {total_epochs}")

            map50_col = [c for c in df.columns if "mAP50(B)" in c or "mAP50" in c]
            map50_95_col = [c for c in df.columns if "mAP50-95(B)" in c or "mAP50-95" in c]

            if map50_col:
                val50 = float(last_row[map50_col[0]])
                self.card_map50_val.configure(text=f"{val50:.4f}")
                best_map50 = df[map50_col[0]].max()
                best_idx = df[map50_col[0]].idxmax()
                best_epoch = int(df['epoch'].iloc[best_idx]) if 'epoch' in df.columns else (best_idx + 1)
                self.card_best_map50_val.configure(text=f"{best_map50:.4f}")
                self.card_best_epoch_val.configure(text=str(best_epoch))

            if map50_95_col:
                val50_95 = float(last_row[map50_95_col[0]])
                self.card_map50_95_val.configure(text=f"{val50_95:.4f}")

            train_loss_cols = [c for c in df.columns if "loss" in c.lower() and "val" not in c.lower()]
            if train_loss_cols:
                total_loss = sum(float(last_row[c]) for c in train_loss_cols)
                self.card_loss_val.configure(text=f"{total_loss:.4f}")
            else:
                loss_any = [c for c in df.columns if "loss" in c.lower()]
                if loss_any:
                    self.card_loss_val.configure(text=f"{float(last_row[loss_any[0]]):.4f}")

            lr_cols = [c for c in df.columns if "lr" in c.lower()]
            if lr_cols:
                curr_lr = float(last_row[lr_cols[0]])
                self.card_lr_val.configure(text=f"{curr_lr:.6f}")

            gpu_util, vram_usage = self.get_gpu_telemetry()
            self.card_gpu_util_val.configure(text=gpu_util)
            self.card_vram_val.configure(text=vram_usage)

        except Exception as e:
            print(f"[Telemetry Warning] Error updating cards: {e}")

    def open_analytics_popup(self):
        if not self.last_run_dir or not os.path.exists(self.last_run_dir):
            messagebox.showerror("Error", "No training directory loaded yet. Run a training session or load a results.csv first.")
            return

        popup = ctk.CTkToplevel(self)
        popup.title("Advanced YOLO Diagnostic Curves & Reports")
        popup.geometry("950x750")
        popup.attributes("-topmost", True)

        tabview = ctk.CTkTabview(popup)
        tabview.pack(fill="both", expand=True, padx=15, pady=15)

        tab_cm = tabview.add("Confusion Matrix")
        tab_p = tabview.add("Box Precision (BoxP)")
        tab_r = tabview.add("Box Recall (BoxR)")
        tab_f1 = tabview.add("Box F1 Curve (BoxF1)")
        tab_labels = tabview.add("Labels Summary")
        tab_results = tabview.add("Results Overview")

        self.render_image_on_tab(tab_cm, os.path.join(self.last_run_dir, "confusion_matrix.png"))
        self.render_image_on_tab(tab_p, os.path.join(self.last_run_dir, "BoxP_curve.png"))
        self.render_image_on_tab(tab_r, os.path.join(self.last_run_dir, "BoxR_curve.png"))
        self.render_image_on_tab(tab_f1, os.path.join(self.last_run_dir, "BoxF1_curve.png"))
        self.render_image_on_tab(tab_labels, os.path.join(self.last_run_dir, "labels.jpg"))
        self.render_image_on_tab(tab_results, os.path.join(self.last_run_dir, "results.png"))

    def render_image_on_tab(self, tab, image_path):
        if os.path.exists(image_path):
            img = Image.open(image_path)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(750, 550))
            lbl = ctk.CTkLabel(tab, image=ctk_img, text="")
            lbl.pack(fill="both", expand=True, padx=10, pady=10)
        else:
            lbl = ctk.CTkLabel(tab, text=f"File not found:\n{os.path.basename(image_path)}\n\n(This diagnostic image generates automatically after full training completion.)")
            lbl.pack(fill="both", expand=True, padx=10, pady=10)

    def export_pdf_report(self):
        if not self.last_run_dir or not os.path.exists(self.last_run_dir):
            messagebox.showerror("Error", "No active or loaded training run found to generate a report.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Documents", "*.pdf")])
        if not save_path:
            return

        try:
            doc = SimpleDocTemplate(save_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#1e293b'), spaceAfter=12)
            story.append(Paragraph("YOLO26 Model Training Summary Report", title_style))
            story.append(Spacer(1, 10))

            csv_path = os.path.join(self.last_run_dir, "results.csv")
            final_map50 = "N/A"
            final_map50_95 = "N/A"
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                map50_col = [c for c in df.columns if "mAP50" in c]
                map50_95_col = [c for c in df.columns if "mAP50-95" in c]
                if map50_col:
                    final_map50 = f"{df[map50_col[0]].iloc[-1]:.4f}"
                if map50_95_col:
                    final_map50_95 = f"{df[map50_95_col[0]].iloc[-1]:.4f}"

            selected_classes = self.get_selected_classes()
            if selected_classes is None:
                classes_kept_val = "All Classes"
            else:
                classes_kept_val = ", ".join([f"[{i}] {self.class_names.get(i, str(i))}" for i in selected_classes])
            if not classes_kept_val:
                classes_kept_val = "None (0 selected)"

            strat = self.freeze_strategy_var.get()
            if strat == "Two-Stage Transfer Learning":
                strat_display = f"Two-Stage (S1: {int(float(self.stage1_split_slider.get()))}%, S2 LR: {self.stage2_lr_scale_entry.get()}x)"
            elif strat == "Custom":
                strat_display = f"Custom Freeze ({self.custom_freeze_entry.get()} layers)"
            else:
                strat_display = strat

            batch_mode = self.batch_option.get()
            batch_display = self.batch_custom_entry.get() if batch_mode == "Custom" else batch_mode

            summary_data = [
                ["Metric / Setting", "Configured Value"],
                ["Dataset YAML", os.path.basename(self.yaml_entry.get()) if self.yaml_entry.get() else "Custom/Loaded Run"],
                ["Classes Trained", classes_kept_val],
                ["Starting Base Model", self.weights_option.get()],
                ["Training Strategy", strat_display],
                ["Total Epochs", str(int(self.epochs_slider.get()))],
                ["Batch / Image Size", f"{batch_display} / {self.imgsz_option.get()}"],
                ["Optimizer Engine", self.adv_opt_option.get()],
                ["Learning Rates (lr0 / lrf)", f"{self.adv_lr0_entry.get()} / {self.adv_lrf_entry.get()}"],
                ["Momentum / Weight Decay", f"{self.adv_momentum_entry.get()} / {self.adv_wd_entry.get()}"],
                ["Warmup Epochs", self.adv_warmup_entry.get()],
                ["Loss Gains (box / cls / dfl)", f"{self.adv_box_entry.get()} / {self.adv_cls_entry.get()} / {self.adv_dfl_entry.get()}"],
                ["Patience (Early Stop)", self.patience_entry.get() if self.patience_switch.get() == 1 else "Disabled (0)"],
                ["Final mAP@50", final_map50],
                ["Final mAP@50-95", final_map50_95],
            ]

            t = Table(summary_data, colWidths=[220, 280])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
            ]))
            story.append(t)
            story.append(Spacer(1, 15))

            aug_data = [
                ["Augmentation Parameter", "Configured Setting"],
                ["Horizontal Flip (fliplr)", "Enabled (p=0.5)" if self.aug_fliplr_switch.get() == 1 else "Disabled (p=0.0)"],
                ["Vertical Flip (flipud)", "Enabled (p=0.5)" if self.aug_flipud_switch.get() == 1 else "Disabled (p=0.0)"],
                ["Rotation (degrees)", f"±{float(self.aug_degrees_slider.get()):.1f}°"],
                ["Scale (scale)", f"±{float(self.aug_scale_slider.get()):.2f}"],
                ["Translation (translate)", f"±{float(self.aug_translate_slider.get()):.2f}"],
                ["HSV-Hue (hsv_h)", f"±{float(self.aug_hsv_h_slider.get()):.3f}"],
                ["HSV-Saturation (hsv_s)", f"±{float(self.aug_hsv_s_slider.get()):.2f}"],
                ["HSV-Value (hsv_v)", f"±{float(self.aug_hsv_v_slider.get()):.2f}"],
                ["Mosaic (4-image mix)", "Enabled (p=1.0)" if self.aug_mosaic_switch.get() == 1 else "Disabled (p=0.0)"],
                ["MixUp", "Enabled (p=0.15)" if self.aug_mixup_switch.get() == 1 else "Disabled (p=0.0)"],
                ["Copy-Paste", "Enabled (p=0.30)" if self.aug_copypaste_switch.get() == 1 else "Disabled (p=0.0)"],
            ]

            story.append(Paragraph("Data Augmentation Parameters", styles['Heading2']))
            story.append(Spacer(1, 5))

            aug_table = Table(aug_data, colWidths=[220, 280])
            aug_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
            ]))
            story.append(aug_table)
            
            story.append(PageBreak())

            results_png = os.path.join(self.last_run_dir, "results.png")
            cm_png = os.path.join(self.last_run_dir, "confusion_matrix.png")
            box_f1_png = os.path.join(self.last_run_dir, "BoxF1_curve.png")
            box_p_png = os.path.join(self.last_run_dir, "BoxP_curve.png")
            box_pr_png = os.path.join(self.last_run_dir, "BoxPR_curve.png")
            box_r_png = os.path.join(self.last_run_dir, "BoxR_curve.png")
            labels_jpg = os.path.join(self.last_run_dir, "labels.jpg")

            heading_style = ParagraphStyle(
                'BoundHeading2',
                parent=styles['Heading2'],
                keepWithNext=True,
                spaceAfter=6
            )

            def add_image_section(title_text, img_path, width, height):
                if os.path.exists(img_path):
                    story.append(KeepTogether([
                        Paragraph(title_text, heading_style),
                        Spacer(1, 4),
                        ReportLabImage(img_path, width=width, height=height),
                        Spacer(1, 14)
                    ]))

            add_image_section("Training Loss & Performance Curves", results_png, width=480, height=240)
            add_image_section("Validation Confusion Matrix", cm_png, width=380, height=270)
            add_image_section("Precision-Recall Curve (BoxPR)", box_pr_png, width=380, height=270)
            add_image_section("F1 Confidence Curve (BoxF1)", box_f1_png, width=380, height=270)
            add_image_section("Precision Curve (BoxP)", box_p_png, width=380, height=270)
            add_image_section("Recall Curve (BoxR)", box_r_png, width=380, height=270)
            add_image_section("Dataset Labels Summary", labels_jpg, width=480, height=330)

            doc.build(story)
            messagebox.showinfo("Export Successful", f"PDF Summary Report saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to generate PDF report: {str(e)}")

    def setup_prediction_tab(self):
        self.tab_predict.grid_columnconfigure(0, weight=1)
        self.tab_predict.grid_columnconfigure(1, weight=1)

        control_frame = ctk.CTkFrame(self.tab_predict)
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(control_frame, text="Prediction Configuration", font=ctk.CTkFont(size=15, weight="bold")).pack(anchor="w", padx=15, pady=10)

        self.custom_weights_entry = ctk.CTkEntry(control_frame, placeholder_text="Path to best.pt")
        self.custom_weights_entry.pack(fill="x", padx=15, pady=5)
        ctk.CTkButton(control_frame, text="Browse Weights", command=self.browse_weights, fg_color="gray").pack(anchor="e", padx=15, pady=2)

        self.source_entry = ctk.CTkEntry(control_frame, placeholder_text="Path to test image")
        self.source_entry.pack(fill="x", padx=15, pady=10)
        ctk.CTkButton(control_frame, text="Select Image", command=self.browse_image, fg_color="gray").pack(anchor="e", padx=15, pady=2)

        ctk.CTkButton(
            control_frame, text="Detect Objects", font=ctk.CTkFont(size=14, weight="bold"), command=self.run_inference
        ).pack(fill="x", padx=15, pady=25)

        display_frame = ctk.CTkFrame(self.tab_predict)
        display_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.img_label = ctk.CTkLabel(display_frame, text="Inference image output will be rendered here.")
        self.img_label.pack(fill="both", expand=True, padx=15, pady=15)

    def setup_export_tab(self):
        self.tab_export.grid_columnconfigure(0, weight=1)

        export_card = ctk.CTkFrame(self.tab_export)
        export_card.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(export_card, text="Model Export Settings", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))

        self.export_weights_entry = ctk.CTkEntry(export_card, placeholder_text="Path to target .pt model file")
        self.export_weights_entry.pack(fill="x", padx=20, pady=5)
        ctk.CTkButton(export_card, text="Browse Weights", command=self.browse_export_weights, fg_color="gray").pack(anchor="e", padx=20, pady=2)

        ctk.CTkLabel(export_card, text="Export Format:").pack(anchor="w", padx=20, pady=(15, 2))
        self.export_format_option = ctk.CTkOptionMenu(
            export_card, 
            values=["onnx", "engine", "tflite", "openvino", "coreml", "pb"]
        )
        self.export_format_option.set("onnx")
        self.export_format_option.pack(anchor="w", padx=20, pady=5)

        ctk.CTkLabel(export_card, text="Export Image Size (imgsz):").pack(anchor="w", padx=20, pady=(15, 2))
        self.export_imgsz = ctk.CTkSegmentedButton(export_card, values=["416", "640", "1024"])
        self.export_imgsz.set("640")
        self.export_imgsz.pack(anchor="w", padx=20, pady=5)

        self.dynamic_switch = ctk.CTkSwitch(export_card, text="Enable Dynamic Batch Size")
        self.dynamic_switch.pack(anchor="w", padx=20, pady=10)

        self.half_switch = ctk.CTkSwitch(export_card, text="Use Half Precision (FP16)")
        self.half_switch.pack(anchor="w", padx=20, pady=5)

        self.export_btn = ctk.CTkButton(
            export_card, text="Export Model", font=ctk.CTkFont(size=15, weight="bold"), command=self.start_export_thread
        )
        self.export_btn.pack(anchor="w", padx=20, pady=30)

        self.export_log = ctk.CTkLabel(export_card, text="Status: Ready", font=ctk.CTkFont(size=13))
        self.export_log.pack(anchor="w", padx=20, pady=5)

    def browse_yaml(self):
        filename = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
        if filename:
            self.yaml_entry.delete(0, "end")
            self.yaml_entry.insert(0, filename)
            self.ds_yaml_entry.delete(0, "end")
            self.ds_yaml_entry.insert(0, filename)
            self.update_class_checkboxes(filename)

    def browse_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.project_entry.delete(0, "end")
            self.project_entry.insert(0, directory)

    def browse_weights(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt")])
        if filename:
            self.custom_weights_entry.delete(0, "end")
            self.custom_weights_entry.insert(0, filename)

    def browse_export_weights(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt")])
        if filename:
            self.export_weights_entry.delete(0, "end")
            self.export_weights_entry.insert(0, filename)

    def browse_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if filename:
            self.source_entry.delete(0, "end")
            self.source_entry.insert(0, filename)

    def start_training_thread(self):
        yaml_path = self.yaml_entry.get()
        if not os.path.exists(yaml_path):
            messagebox.showerror("Error", "Please specify a valid data.yaml file.")
            return

        selected_classes = self.get_selected_classes()
        if selected_classes is not None and len(selected_classes) == 0:
            messagebox.showerror("Class Selection Error", "You must select at least one class to train on, or select 'All'.")
            return

        weights_choice = self.weights_option.get()
        if weights_choice == "[Custom Weights...]":
            messagebox.showerror("Starting Weights Error", "Please select a valid pretrained weight or custom .pt file.")
            return
        if weights_choice.endswith(".pt") and os.path.isabs(weights_choice) and not os.path.exists(weights_choice):
            messagebox.showerror("Starting Weights Error", f"Specified weight file was not found:\n{weights_choice}")
            return

        try:
            self.get_batch_size()
        except ValueError as ve:
            messagebox.showerror("Invalid Batch Size", str(ve))
            return

        try:
            float(self.adv_lr0_entry.get())
            float(self.adv_lrf_entry.get())
            float(self.adv_momentum_entry.get())
            float(self.adv_wd_entry.get())
            float(self.adv_warmup_entry.get())
            float(self.adv_box_entry.get())
            float(self.adv_cls_entry.get())
            float(self.adv_dfl_entry.get())
            if self.patience_switch.get() == 1:
                int(self.patience_entry.get())

            strat = self.freeze_strategy_var.get()
            if strat == "Custom":
                int(self.custom_freeze_entry.get())
            elif strat == "Two-Stage Transfer Learning":
                scale = float(self.stage2_lr_scale_entry.get())
                if scale <= 0:
                    raise ValueError("Stage 2 LR Multiplier must be greater than 0.")
        except ValueError as ve:
            messagebox.showerror("Invalid Input", f"Please check your input values:\n\n{str(ve)}")
            return

        self.stop_requested = False
        self.pause_event.set()

        self.train_btn.configure(state="disabled", text="Running...")
        self.pause_btn.configure(state="normal", text="|| Pause")
        self.stop_btn.configure(state="normal")
        
        self.card_status_badge.configure(text="TRAINING", text_color="#2ef072")

        self.progress_bar.set(0.0)
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", "end")
        self.console_text.configure(state="disabled")
        
        self.start_time = time.time()

        threading.Thread(target=self.run_training, daemon=True).start()

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_btn.configure(text="> Resume", fg_color="#22c55e", hover_color="#16a34a")
        else:
            self.pause_event.set()
            self.pause_btn.configure(text="|| Pause", fg_color="#eab308", hover_color="#ca8a04")

    def request_stop(self):
        self.stop_requested = True
        self.stop_btn.configure(state="disabled", text="Stopping...")
        self.pause_event.set()

    def run_training(self):
        redirector_stdout = QueueRedirector(self.log_queue, sys.__stdout__)
        redirector_stderr = QueueRedirector(self.log_queue, sys.__stderr__)
        
        sys.stdout = redirector_stdout
        sys.stderr = redirector_stderr

        for handler in LOGGER.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(redirector_stdout)
        
        try:
            weights = self.weights_option.get()
            self.total_epochs = int(self.epochs_slider.get())
            batch = self.get_batch_size()
            imgsz = int(self.imgsz_option.get())
            yaml_path = self.yaml_entry.get()
            
            project_path = self.project_entry.get().strip()
            if not project_path:
                project_path = "runs/detect"

            opt_val = self.adv_opt_option.get()
            lr0_val = float(self.adv_lr0_entry.get())
            lrf_val = float(self.adv_lrf_entry.get())
            momentum_val = float(self.adv_momentum_entry.get())
            wd_val = float(self.adv_wd_entry.get())
            warmup_val = float(self.adv_warmup_entry.get())
            box_val = float(self.adv_box_entry.get())
            cls_val = float(self.adv_cls_entry.get())
            dfl_val = float(self.adv_dfl_entry.get())

            fliplr_val = 0.5 if self.aug_fliplr_switch.get() == 1 else 0.0
            flipud_val = 0.5 if self.aug_flipud_switch.get() == 1 else 0.0
            degrees_val = float(self.aug_degrees_slider.get())
            scale_val = float(self.aug_scale_slider.get())
            translate_val = float(self.aug_translate_slider.get())
            hsv_h_val = float(self.aug_hsv_h_slider.get())
            hsv_s_val = float(self.aug_hsv_s_slider.get())
            hsv_v_val = float(self.aug_hsv_v_slider.get())
            mosaic_val = 1.0 if self.aug_mosaic_switch.get() == 1 else 0.0
            mixup_val = 0.15 if self.aug_mixup_switch.get() == 1 else 0.0
            copypaste_val = 0.3 if self.aug_copypaste_switch.get() == 1 else 0.0

            if self.patience_switch.get() == 1:
                patience_val = int(self.patience_entry.get())
            else:
                patience_val = 0

            aug_banner = (
                "\n" + "=" * 60 + "\n"
                "[System] YOLO26 ACTIVE DATA AUGMENTATION CONFIGURATION\n"
                + "-" * 60 + "\n"
                f"  Horizontal Flip (fliplr) : {'ENABLED (p=0.5)' if fliplr_val > 0 else 'DISABLED (p=0.0)'}\n"
                f"  Vertical Flip (flipud)   : {'ENABLED (p=0.5)' if flipud_val > 0 else 'DISABLED (p=0.0)'}\n"
                f"  Rotation (degrees)       : +/-{degrees_val:.1f} deg\n"
                f"  Scale (scale)            : +/-{scale_val:.2f}\n"
                f"  Translation (translate)  : +/-{translate_val:.2f}\n"
                f"  HSV-Hue (hsv_h)          : +/-{hsv_h_val:.3f}\n"
                f"  HSV-Saturation (hsv_s)   : +/-{hsv_s_val:.2f}\n"
                f"  HSV-Value (hsv_v)        : +/-{hsv_v_val:.2f}\n"
                f"  Mosaic (4-image mix)     : {'ENABLED (p=1.0)' if mosaic_val > 0 else 'DISABLED (p=0.0)'}\n"
                f"  MixUp                    : {'ENABLED (p=0.15)' if mixup_val > 0 else 'DISABLED (p=0.0)'}\n"
                f"  Copy-Paste               : {'ENABLED (p=0.30)' if copypaste_val > 0 else 'DISABLED (p=0.0)'}\n"
                + "=" * 60 + "\n"
            )
            print(aug_banner)

            selected_strategy = self.freeze_strategy_var.get()

            base_kwargs = {
                "data": yaml_path,
                "batch": batch,
                "imgsz": imgsz,
                "optimizer": opt_val,
                "lrf": lrf_val,
                "momentum": momentum_val,
                "weight_decay": wd_val,
                "warmup_epochs": warmup_val,
                "box": box_val,
                "cls": cls_val,
                "dfl": dfl_val,
                "fliplr": fliplr_val,
                "flipud": flipud_val,
                "degrees": degrees_val,
                "scale": scale_val,
                "translate": translate_val,
                "hsv_h": hsv_h_val,
                "hsv_s": hsv_s_val,
                "hsv_v": hsv_v_val,
                "mosaic": mosaic_val,
                "mixup": mixup_val,
                "copy_paste": copypaste_val,
                "patience": patience_val,
                "project": project_path,
                "device": 0 if torch.cuda.is_available() else "cpu",
                "compile": "max-autotune",
                "cache": "disk",
                "amp": "bf16",
            }

            selected_classes = self.get_selected_classes()
            if selected_classes is not None:
                base_kwargs["classes"] = selected_classes

            def on_train_batch_end(trainer):
                if self.stop_requested:
                    trainer.stop = True
                    self.log_queue.put("\n[System] Training interrupted by user. Saving current state...\n")
                
                if not self.pause_event.is_set():
                    self.log_queue.put("\n[System] Training Paused...\n")
                    self.pause_event.wait()
                    self.log_queue.put("\n[System] Training Resumed...\n")

            if selected_strategy == "Two-Stage Transfer Learning":
                stage1_pct = float(self.stage1_split_slider.get()) / 100.0
                stage1_epochs = max(1, int(self.total_epochs * stage1_pct))
                stage2_epochs = max(1, self.total_epochs - stage1_epochs)
                stage2_lr_scale = float(self.stage2_lr_scale_entry.get())
                stage2_lr0 = lr0_val * stage2_lr_scale

                self.log_queue.put(f"\n{'='*65}\nLAUNCHING STAGE 1/2: FROZEN BACKBONE (Layers 0-9 Locked)\n"
                                   f"Budget: {stage1_epochs} Epochs | Initial LR: {lr0_val}\n{'='*65}\n")
                
                model_stage1 = YOLO(weights)
                model_stage1.add_callback("on_train_batch_end", on_train_batch_end)

                def on_stage1_epoch_end(trainer):
                    curr = trainer.epoch + 1
                    ratio = curr / self.total_epochs
                    elapsed_sec = time.time() - self.start_time
                    avg = elapsed_sec / curr
                    rem_sec = avg * (self.total_epochs - curr)
                    self.after(0, lambda: self.update_progress_ui(
                        f"Stage 1: {curr}/{stage1_epochs}", ratio,
                        time.strftime("%H:%M:%S", time.gmtime(elapsed_sec)),
                        time.strftime("%H:%M:%S", time.gmtime(rem_sec))
                    ))
                    self.last_run_dir = str(trainer.save_dir)
                    live_csv = os.path.join(self.last_run_dir, "results.csv")
                    if os.path.exists(live_csv):
                        self.after(0, lambda p=live_csv: self.plot_csv_metrics(p, silent=True))

                model_stage1.add_callback("on_train_epoch_end", on_stage1_epoch_end)

                s1_kwargs = dict(base_kwargs)
                s1_kwargs.update({
                    "epochs": stage1_epochs,
                    "lr0": lr0_val,
                    "freeze": 10,
                    "name": "stage1_frozen_run"
                })
                s1_results = model_stage1.train(**s1_kwargs)
                self.last_run_dir = str(s1_results.save_dir)

                if self.stop_requested:
                    messagebox.showinfo("Stopped", "Training was stopped safely during Stage 1.")
                    return

                stage1_best = os.path.join(self.last_run_dir, "weights", "best.pt")
                if not os.path.exists(stage1_best):
                    stage1_best = os.path.join(self.last_run_dir, "weights", "last.pt")

                self.log_queue.put(f"\n{'='*65}\nLAUNCHING STAGE 2/2: FULL FINE-TUNING (All Layers Unlocked)\n"
                                   f"Base Weights: {os.path.basename(stage1_best)}\n"
                                   f"Budget: {stage2_epochs} Epochs | Reduced LR: {stage2_lr0:.6f} (Scale: {stage2_lr_scale}x)\n{'='*65}\n")

                model_stage2 = YOLO(stage1_best)
                model_stage2.add_callback("on_train_batch_end", on_train_batch_end)

                def on_stage2_epoch_end(trainer):
                    curr = trainer.epoch + 1
                    total_done = stage1_epochs + curr
                    ratio = total_done / self.total_epochs
                    elapsed_sec = time.time() - self.start_time
                    avg = elapsed_sec / total_done
                    rem_sec = avg * (self.total_epochs - total_done)
                    self.after(0, lambda: self.update_progress_ui(
                        f"Stage 2: {curr}/{stage2_epochs}", ratio,
                        time.strftime("%H:%M:%S", time.gmtime(elapsed_sec)),
                        time.strftime("%H:%M:%S", time.gmtime(rem_sec))
                    ))
                    self.last_run_dir = str(trainer.save_dir)
                    live_csv = os.path.join(self.last_run_dir, "results.csv")
                    if os.path.exists(live_csv):
                        self.after(0, lambda p=live_csv: self.plot_csv_metrics(p, silent=True))

                model_stage2.add_callback("on_train_epoch_end", on_stage2_epoch_end)

                s2_kwargs = dict(base_kwargs)
                s2_kwargs.update({
                    "epochs": stage2_epochs,
                    "lr0": stage2_lr0,
                    "freeze": 0,
                    "name": "stage2_finetune_run"
                })
                s2_results = model_stage2.train(**s2_kwargs)
                self.last_run_dir = str(s2_results.save_dir)

            else:
                if selected_strategy == "Full Fine-Tuning":
                    freeze_val = 0
                elif selected_strategy == "Freeze Backbone":
                    freeze_val = 10
                elif selected_strategy == "Freeze Backbone + Neck":
                    freeze_val = 22
                elif selected_strategy == "Custom":
                    freeze_val = int(self.custom_freeze_entry.get())
                else:
                    freeze_val = 0

                model = YOLO(weights)
                model.add_callback("on_train_batch_end", on_train_batch_end)

                def on_train_epoch_end(trainer):
                    current_epoch = trainer.epoch + 1
                    progress_ratio = current_epoch / self.total_epochs
                    elapsed_sec = time.time() - self.start_time
                    avg_time = elapsed_sec / current_epoch
                    remaining_sec = avg_time * (self.total_epochs - current_epoch)
                    self.after(0, lambda: self.update_progress_ui(
                        f"{current_epoch} / {self.total_epochs}", progress_ratio,
                        time.strftime("%H:%M:%S", time.gmtime(elapsed_sec)),
                        time.strftime("%H:%M:%S", time.gmtime(remaining_sec))
                    ))
                    self.last_run_dir = str(trainer.save_dir)
                    live_csv = os.path.join(self.last_run_dir, "results.csv")
                    if os.path.exists(live_csv):
                        self.after(0, lambda p=live_csv: self.plot_csv_metrics(p, silent=True))

                model.add_callback("on_train_epoch_end", on_train_epoch_end)

                train_kwargs = dict(base_kwargs)
                train_kwargs.update({
                    "epochs": self.total_epochs,
                    "lr0": lr0_val,
                    "freeze": freeze_val,
                    "name": "gui_train_run"
                })

                results = model.train(**train_kwargs)
                self.last_run_dir = str(results.save_dir)

            csv_path = os.path.join(self.last_run_dir, "results.csv")
            if os.path.exists(csv_path):
                self.plot_csv_metrics(csv_path)

            if not self.stop_requested:
                messagebox.showinfo("Success", "Training completed successfully!")
            else:
                messagebox.showinfo("Stopped", "Training was stopped safely. Progress has been saved.")

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            for handler in LOGGER.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setStream(sys.__stdout__)
                    
            self.train_btn.configure(state="normal", text="Launch")
            self.pause_btn.configure(state="disabled", text="|| Pause", fg_color="#eab308")
            self.stop_btn.configure(state="disabled", text="[ ] Stop")
            self.card_status_badge.configure(text="FINISHED", text_color="#94a3b8")

    def update_progress_ui(self, epoch_str, progress_ratio, elapsed_str, eta_str):
        self.progress_bar.set(progress_ratio)
        percent = int(progress_ratio * 100)
        self.progress_text.configure(text=f"Progress: {epoch_str} ({percent}%)")
        self.elapsed_label.configure(text=f"Elapsed: {elapsed_str}")
        self.eta_label.configure(text=f"ETA: {eta_str}")

    def load_results_csv(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.last_run_dir = os.path.dirname(filename)
            self.plot_csv_metrics(filename)

    def plot_csv_metrics(self, csv_path, silent=False):
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            self.ax.clear()
            self.ax.grid(True, linestyle="--", alpha=0.3)

            map50_col = [c for c in df.columns if "metrics/mAP50(B)" in c or "mAP50" in c]
            map50_95_col = [c for c in df.columns if "metrics/mAP50-95(B)" in c or "mAP50-95" in c]

            if map50_col and map50_95_col:
                epochs = df['epoch'] if 'epoch' in df.columns else df.index
                self.ax.plot(epochs, df[map50_col[0]], label="mAP@50", color="#2ef072", linewidth=2)
                self.ax.plot(epochs, df[map50_95_col[0]], label="mAP@50-95", color="#3b82f6", linewidth=2)

                self.ax.set_title("mAP Convergence Curve", color="white")
                self.ax.set_xlabel("Epochs")
                self.ax.set_ylabel("Score")
                self.ax.legend(facecolor="#2b2b2b", edgecolor="none", labelcolor="white")
                self.chart_canvas.draw()

                self.metrics_status.configure(text=f"Loaded: {os.path.basename(csv_path)}")
                self.update_live_metrics_cards(df)
            else:
                if not silent:
                    messagebox.showwarning("Plotting Warning", "Could not identify standard mAP metric columns in the CSV.")
        except Exception as e:
            if not silent:
                messagebox.showerror("Plot Error", f"Failed to parse CSV file: {str(e)}")

    def run_inference(self):
        weights_path = self.custom_weights_entry.get()
        image_path = self.source_entry.get()

        if not os.path.exists(weights_path) or not os.path.exists(image_path):
            messagebox.showerror("Error", "Valid custom model weights and image path are required.")
            return

        try:
            model = YOLO(weights_path)
            results = model.predict(source=image_path, device=0 if torch.cuda.is_available() else "cpu")

            result = results[0]
            rgb_image = Image.fromarray(result.plot()[:, :, ::-1])

            ctk_img = ctk.CTkImage(light_image=rgb_image, dark_image=rgb_image, size=(480, 480))
            self.img_label.configure(image=ctk_img, text="")
        except Exception as e:
            messagebox.showerror("Inference Error", str(e))

    def start_export_thread(self):
        weights_path = self.export_weights_entry.get()
        if not os.path.exists(weights_path):
            messagebox.showerror("Error", "Please select a valid .pt weights file to export.")
            return

        fmt = self.export_format_option.get().upper()
        self.export_btn.configure(state="disabled", text=f"Exporting to {fmt}...")
        self.export_log.configure(text=f"Status: Converting PyTorch weights to {fmt}...")
        
        threading.Thread(target=self.run_export, daemon=True).start()

    def run_export(self):
        try:
            weights_path = self.export_weights_entry.get()
            imgsz = int(self.export_imgsz.get())
            dynamic = bool(self.dynamic_switch.get())
            half = bool(self.half_switch.get())
            export_format = self.export_format_option.get()

            model = YOLO(weights_path)
            
            exported_path = model.export(
                format=export_format,
                imgsz=imgsz,
                dynamic=dynamic,
                half=half,
                device=0 if torch.cuda.is_available() else "cpu"
            )

            self.export_log.configure(text=f"Status: Successfully saved to {os.path.basename(str(exported_path))}")
            messagebox.showinfo("Export Complete", f"{export_format.upper()} Model exported successfully!\n\nLocation:\n{exported_path}")
        except Exception as e:
            self.export_log.configure(text="Status: Export Failed")
            messagebox.showerror("Export Error", str(e))
        finally:
            self.export_btn.configure(state="normal", text="Export Model")

if __name__ == "__main__":
    app = YOLO26App()
    app.mainloop()