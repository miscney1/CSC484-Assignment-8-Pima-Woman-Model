"""
Pima Woman Predictive Model - CSC484 Portfolio Project

A small educational neural network trained only on the assigned Pima Indians
Diabetes dataset. The interface lets the user build a hypothetical woman,
move her through the 1965-2007 study period, and see how the trained model
responds to the selected measurements.

This program is for education only. The percentage is model output, not a
clinical diagnosis or validated medical risk estimate.
"""

import random
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np

# 1. PROJECT SETTINGS
BASE = Path(__file__).resolve().parent
DATA_NAMES = ["pima_diabetes.csv", "pima-indians-diabetes.data.csv"]
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "Pedigree", "Age"
]
START_YEAR, END_YEAR, SEED = 1965, 2007, 42

# These values are filled when the model is trained.
MODEL = {}
STATS = {}
PEDIGREE_LEVELS = [0.25, 0.50, 0.75]

# 2. LOAD AND PREPARE THE ASSIGNED PIMA DATA
def find_data_file():
    # Resolve the dataset from the program's own folder, not the folder used
    # to launch Python. This keeps the project portable when cloned from GitHub.
    candidates = [
        BASE / "pima_diabetes.csv",
        BASE / "pima-indians-diabetes.data.csv",
        BASE / "data" / "pima_diabetes.csv",
        BASE / "data" / "pima-indians-diabetes.data.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            sample = np.loadtxt(path, delimiter=",")
            if sample.ndim == 2 and sample.shape[1] == 9 and len(sample) >= 700:
                return path, sample
        except (ValueError, OSError):
            pass
    expected = BASE / "pima_diabetes.csv"
    raise FileNotFoundError(
        f"Pima dataset not found. Expected: {expected}\n"
        "Keep pima_diabetes.csv in the same folder as this Python program."
    )

def load_data():
    data_file, data = find_data_file()
    x = data[:, :8].astype(float)
    y = data[:, 8].astype(float).reshape(-1, 1)

    # Zero is not a realistic value for these measurements. In the original
    # classroom file, zeros are commonly treated as missing measurements.
    for col in [1, 2, 3, 4, 5]:
        x[x[:, col] == 0, col] = np.nan
    return x, y, data_file

def split_data(x, y):
    rng = np.random.default_rng(SEED)
    negative = np.where(y.ravel() == 0)[0]
    positive = np.where(y.ravel() == 1)[0]
    rng.shuffle(negative)
    rng.shuffle(positive)
    n0, n1 = int(len(negative) * 0.80), int(len(positive) * 0.80)
    train_idx = np.concatenate([negative[:n0], positive[:n1]])
    test_idx = np.concatenate([negative[n0:], positive[n1:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx

def prepare_data(x, train_idx):
    medians = np.nanmedian(x[train_idx], axis=0)
    clean = np.where(np.isnan(x), medians, x)
    means = clean[train_idx].mean(axis=0)
    stds = clean[train_idx].std(axis=0)
    stds[stds == 0] = 1
    return (clean - means) / stds, medians, means, stds

# 3. DEFINE, FIT, EVALUATE, AND USE A SMALL NEURAL NETWORK
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def train_model():
    """Train an 8 -> 12 -> 1 feed-forward neural network with gradient descent."""
    global MODEL, STATS, PEDIGREE_LEVELS
    x, y, data_file = load_data()
    train_idx, test_idx = split_data(x, y)
    x_scaled, medians, means, stds = prepare_data(x, train_idx)
    x_train, y_train = x_scaled[train_idx], y[train_idx]
    x_test, y_test = x_scaled[test_idx], y[test_idx]

    # Map the unfamiliar pedigree score to dataset quartiles for the interface.
    pedigree = x[:, 6][~np.isnan(x[:, 6])]
    PEDIGREE_LEVELS = np.quantile(pedigree, [0.25, 0.50, 0.75]).tolist()

    rng = np.random.default_rng(SEED)
    w1 = rng.normal(0, 0.30, (8, 12))
    b1 = np.zeros((1, 12))
    w2 = rng.normal(0, 0.30, (12, 1))
    b2 = np.zeros((1, 1))
    learning_rate = 0.02
    epochs = 1500

    # Forward pass, measure error, backpropagate, then adjust weights.
    for _ in range(epochs):
        z1 = x_train @ w1 + b1
        hidden = np.maximum(z1, 0)                      # ReLU hidden layer
        probs = sigmoid(hidden @ w2 + b2)              # Sigmoid output layer

        output_error = (probs - y_train) / len(y_train)
        dw2 = hidden.T @ output_error
        db2 = output_error.sum(axis=0, keepdims=True)
        hidden_error = (output_error @ w2.T) * (z1 > 0)
        dw1 = x_train.T @ hidden_error
        db1 = hidden_error.sum(axis=0, keepdims=True)

        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

    test_probs = sigmoid(np.maximum(x_test @ w1 + b1, 0) @ w2 + b2)
    predictions = (test_probs >= 0.50).astype(float)
    accuracy = float((predictions == y_test).mean())

    MODEL = {"w1": w1, "b1": b1, "w2": w2, "b2": b2,
             "medians": medians, "means": means, "stds": stds}
    STATS = {"rows": len(x), "train": len(train_idx), "test": len(test_idx),
             "accuracy": accuracy, "epochs": epochs, "file": data_file.name}

def predict_probability(values):
    row = np.array(values, dtype=float)
    row = np.where(np.isnan(row), MODEL["medians"], row)
    row = (row - MODEL["means"]) / MODEL["stds"]
    hidden = np.maximum(row @ MODEL["w1"] + MODEL["b1"], 0)
    return float(sigmoid(hidden @ MODEL["w2"] + MODEL["b2"]).ravel()[0])

def bmi(height_in, weight_lb):
    return 703 * weight_lb / (height_in ** 2)

# 4. TKINTER USER INTERFACE
class App:
    def __init__(self, root):
        self.root = root
        self.result_ready = False
        self.target = 0.0
        self.anim_step = 0
        root.title("Pima Woman Predictive Model")
        root.geometry("1160x760")
        root.minsize(1000, 700)
        self._styles()

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Pima Woman Predictive Model", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Build a hypothetical woman and see how a neural network trained on the assigned Pima dataset responds.",
            style="Sub.TLabel"
        ).pack(anchor="w", pady=(0, 10))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self.left = ttk.Frame(body, style="Card.TFrame", padding=14)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.right = ttk.Frame(body, style="Card.TFrame", padding=14)
        self.right.grid(row=0, column=1, sticky="nsew")

        self.vars, self.value_labels = {}, {}
        self._build_inputs()
        self._build_results()
        self.status = tk.StringVar(value="Training the neural network on the assigned Pima data...")
        ttk.Label(outer, textvariable=self.status, style="Sub.TLabel").pack(anchor="w", pady=(8, 0))

        # Train once at startup, then reuse those learned weights for each subject.
        try:
            train_model()
            self._training_done()
        except Exception as exc:
            messagebox.showerror("Startup Error", str(exc))
            self.status.set("Model could not be trained.")

        self.root.after(500, self._idle_motion)

    def _styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#edf2f0")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#edf2f0", foreground="#18382f", font=("Segoe UI", 20, "bold"))
        style.configure("Sub.TLabel", background="#edf2f0", foreground="#52645e", font=("Segoe UI", 9))
        style.configure("Head.TLabel", background="#ffffff", foreground="#214b3f", font=("Segoe UI", 11, "bold"))
        style.configure("Card.TLabel", background="#ffffff", foreground="#27342f", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)

    def _scale(self, key, label, low, high, default, note="", formatter=None):
        ttk.Label(self.left, text=label, style="Card.TLabel").pack(anchor="w", pady=(4, 0))
        row = ttk.Frame(self.left, style="Card.TFrame")
        row.pack(fill="x")
        var = tk.DoubleVar(value=default)
        self.vars[key] = var
        ttk.Scale(row, from_=low, to=high, variable=var).pack(side="left", fill="x", expand=True)
        out = ttk.Label(row, width=13, anchor="e", style="Card.TLabel")
        out.pack(side="right", padx=(8, 0))
        self.value_labels[key] = out

        def refresh(*_):
            value = var.get()
            out.config(text=formatter(value) if formatter else f"{value:.0f}")
            self._update_derived()
        var.trace_add("write", refresh)
        refresh()
        if note:
            ttk.Label(self.left, text=note, style="Card.TLabel", foreground="#68766f", wraplength=470).pack(anchor="w")

    def _build_inputs(self):
        ttk.Label(self.left, text="Build the Subject", style="Head.TLabel").pack(anchor="w")
        self.build_btn = ttk.Button(self.left, text="Create Subject", style="Accent.TButton", command=self.create_subject)
        self.build_btn.pack(fill="x", pady=(6, 6))
        self._scale("year", "Study year", START_YEAR, END_YEAR, 1965,
                    "Age advances with the historical study year. Other measurements stay where you set them.")
        self._scale("base_age", "Age in 1965", 21, 39, 28)
        self.age_text = tk.StringVar()
        ttk.Label(self.left, textvariable=self.age_text, style="Card.TLabel").pack(anchor="e")
        self._scale("preg", "Previous pregnancies", 0, 10, 2)
        self._scale("glucose", "2-hour glucose test (mg/dL)", 60, 200, 120,
                    "Blood glucose two hours after an oral glucose drink. This is not A1C.")
        self._scale("dbp", "Diastolic blood pressure (mmHg)", 40, 120, 75,
                    "The lower number in a blood-pressure reading.")
        self._scale("skin", "Triceps skinfold thickness (mm)", 5, 60, 25,
                    "Thickness of a skin-and-fat fold at the back of the upper arm.")
        self._scale("insulin", "2-hour insulin level (µU/mL)", 10, 400, 110)
        self._scale("height", "Height", 55, 74, 64, formatter=lambda x: f"{int(x)//12}'{int(x)%12}\"")
        self._scale("weight", "Weight (lb)", 90, 300, 165)
        self.bmi_text = tk.StringVar()
        ttk.Label(self.left, textvariable=self.bmi_text, style="Card.TLabel").pack(anchor="e")
        self._scale("ped", "Family diabetes history pattern", 0, 2, 1,
                    "Lower / Typical / Higher maps to quartiles of the dataset's pedigree score.",
                    formatter=lambda x: ["Lower", "Typical", "Higher"][round(x)])

        self._update_derived()

    def _build_results(self):
        ttk.Label(self.right, text="Neural Network Result", style="Head.TLabel").pack(anchor="w")
        ttk.Label(self.right, text="STUDENT-TRAINED MODEL USING THE ASSIGNED PIMA DATASET", style="Card.TLabel", foreground="#61726b").pack(anchor="w")
        self.bar = tk.Canvas(self.right, height=64, bg="#f1f4f2", highlightthickness=0)
        self.bar.pack(fill="x", pady=(10, 4))
        self.percent = ttk.Label(self.right, text="Awaiting subject", style="Head.TLabel")
        self.percent.pack(anchor="e")
        ttk.Label(self.right, text="Model probability of the diabetes-positive class. This is not a clinical diagnosis or validated personal risk percentage.", style="Card.TLabel", foreground="#68766f", wraplength=500).pack(anchor="w", pady=(4, 14))

        self.metrics = tk.StringVar(value="Model training pending...")
        ttk.Label(self.right, textvariable=self.metrics, style="Card.TLabel", wraplength=500, justify="left").pack(anchor="w", pady=(4, 16))
        ttk.Separator(self.right).pack(fill="x", pady=8)
        ttk.Label(self.right, text="Data Source", style="Head.TLabel").pack(anchor="w")
        source = (
            "Pima Indians Diabetes Database, the dataset assigned for this CSC484 portfolio project.\n\n"
            "The program uses all eight provided predictors: pregnancies, 2-hour glucose, diastolic blood pressure, triceps skinfold thickness, 2-hour insulin, BMI, diabetes pedigree function, and age.\n\n"
            "The neural network is trained inside this program, evaluated on held-out records, and then used to score the hypothetical subject.\n\n"
            "Historical display: the study-year control runs from 1965 through 2007. It is a scenario tool, not a reconstruction of an individual participant's medical record."
        )
        ttk.Label(self.right, text=source, style="Card.TLabel", wraplength=500, justify="left").pack(anchor="w")

    def _training_done(self):
        self.metrics.set(
            f"Dataset rows: {STATS['rows']}\n"
            f"Training records: {STATS['train']} | Held-out test records: {STATS['test']}\n"
            f"Hidden layer: 12 neurons | Training passes: {STATS['epochs']}\n"
            f"Held-out accuracy: {STATS['accuracy']:.1%}\n"
            f"Loaded data file: {STATS['file']}"
        )
        self.status.set("Model trained. Build the subject, then choose Create Subject.")

    def _update_derived(self):
        if not hasattr(self, "age_text") or "year" not in self.vars or "base_age" not in self.vars:
            return
        age = round(self.vars["base_age"].get()) + round(self.vars["year"].get()) - START_YEAR
        self.age_text.set(f"Current age in selected year: {age}")
        if hasattr(self, "bmi_text") and "height" in self.vars and "weight" in self.vars:
            self.bmi_text.set(f"Calculated BMI: {bmi(self.vars['height'].get(), self.vars['weight'].get()):.1f}")

    def _draw_bar(self, value):
        self.bar.update_idletasks()
        width = max(self.bar.winfo_width(), 420)
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, width, 64, fill="#edf1ef", outline="")
        self.bar.create_rectangle(0, 0, width * max(0, min(1, value)), 64, fill="#477d6c", outline="")

    def _idle_motion(self):
        if not self.result_ready and self.anim_step == 0:
            self._draw_bar(random.uniform(0.18, 0.78))
        self.root.after(650, self._idle_motion)

    def create_subject(self):
        age = round(self.vars["base_age"].get()) + round(self.vars["year"].get()) - START_YEAR
        pedigree = PEDIGREE_LEVELS[round(self.vars["ped"].get())]
        values = [
            self.vars["preg"].get(), self.vars["glucose"].get(), self.vars["dbp"].get(),
            self.vars["skin"].get(), self.vars["insulin"].get(),
            bmi(self.vars["height"].get(), self.vars["weight"].get()), pedigree, age
        ]
        self.target = predict_probability(values)
        self.result_ready = False
        self.anim_step = 1
        self.build_btn.config(state="disabled")
        self.percent.config(text="Calculating...")
        self.status.set("Running the subject through the trained neural network...")
        self._settle_animation()

    def _settle_animation(self):
        progress = min(self.anim_step / 24, 1)
        noise = (1 - progress) * 0.30
        shown = max(0, min(1, self.target + random.uniform(-noise, noise)))
        self._draw_bar(shown)
        if progress < 1:
            self.anim_step += 1
            self.root.after(85 if self.anim_step < 10 else 120, self._settle_animation)
            return
        self.anim_step = 0
        self.result_ready = True
        self._draw_bar(self.target)
        self.percent.config(text=f"{self.target:.1%} diabetes-class probability")
        self.build_btn.config(state="normal", text="Update Subject")
        self.status.set("Subject built. Change any measurement and update the subject to compare another scenario.")

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
