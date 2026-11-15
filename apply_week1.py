"""Week-1 revision patches. Run from the REPO ROOT:  python apply_week1.py
Applies: A5 (grouped inner CV) to notebooks 05 and 05b, the A4 per-subject z-score cell
(flag OFF by default), the notebook-09 markdown fixes, and the CNN-QNN -> MLP-QNN rename.
Every edit is asserted to match exactly once; if anything does not match, NOTHING is written."""
import json, re, subprocess, pathlib, uuid, sys

NB = pathlib.Path("notebooks")

def load(name):
    return json.loads((NB / name).read_text(encoding="utf-8"))

def save(name, nb):
    (NB / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8", newline="\n")

def src(cell):  return "".join(cell["source"])
def setsrc(cell, text): cell["source"] = text.splitlines(keepends=True)

def find_cell(nb, needle, cell_type="code"):
    hits = [c for c in nb["cells"] if c["cell_type"] == cell_type and needle in src(c)]
    if len(hits) != 1:
        raise SystemExit(f"\nSTOPPED (nothing written): looked for the cell containing {needle!r}, "
                         f"found {len(hits)}. Send this message to Claude.")
    return hits[0]

def sub1(text, old, new, label, done=None, soft=False):
    """Replace old->new once. Skip if already applied (new text, or the `done` marker, is present).
    soft=True: cosmetic edit, silently skipped if the old text is not there."""
    if text.count(old) == 1:
        return text.replace(old, new)
    if new in text or (done and done in text):
        print(f"  [skip] {label}: already applied")
        return text
    if soft:
        print(f"  [skip] {label}: text not found (cosmetic)")
        return text
    near = [l.strip() for l in text.splitlines() if old.strip().split("(")[0][:25] in l][:4]
    raise SystemExit(f"\nSTOPPED (nothing written): [{label}] could not find the expected text.\n"
                     f"  expected: {old[:90]!r}\n  lines in your file that look similar: {near}\n"
                     f"Send this message to Claude.")

def new_code_cell(nb, text):
    c = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": text.splitlines(keepends=True)}
    if nb.get("nbformat_minor", 0) >= 5:
        c["id"] = uuid.uuid4().hex[:8]
    return c

# ---------------------------------------------------------------- shared text
IMPORT_OLD = "from sklearn.model_selection import GridSearchCV, StratifiedKFold"
IMPORT_NEW = "from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold"
CV_OLD = "INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)"
CV_NEW = ("# Inner CV is grouped by subject: windows of one subject never sit in both the\n"
          "# inner-train and inner-validation split (fixes leakage in hyperparameter search).\n"
          "INNER_CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)")
PRINT_OLD = "StratifiedKFold(5)"
PRINT_NEW = "StratifiedGroupKFold(5), grouped by subject"

IDX_FN = '''def stratified_subsample_idx(y, n_per_class, random_state=42):
    """Same selection as stratified_subsample(), but returns indices so the
    subject ids (groups) can be subsampled together with X and y."""
    rng = np.random.RandomState(random_state)
    parts = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        parts.append(idx if len(idx) <= n_per_class
                     else rng.choice(idx, size=n_per_class, replace=False))
    allidx = np.concatenate(parts)
    return allidx[rng.permutation(len(allidx))]


'''

SUB_RE = re.compile(
    r"(?P<ind>[ ]+)X_train_sub, y_train_sub = stratified_subsample\(\n"
    r"[ ]+X_train, y_train, n_per_class=n_per_class,\n"
    r"[ ]+random_state=RANDOM_STATE \+ fold_i\)")

def sub_call_repl(m):
    i = m.group("ind")
    return (f"{i}g_train = df.loc[train_mask, subject_col].values   # subject id of each training window\n"
            f"{i}sub_idx = stratified_subsample_idx(\n"
            f"{i}    y_train, n_per_class=n_per_class, random_state=RANDOM_STATE + fold_i)\n"
            f"{i}X_train_sub, y_train_sub, g_train_sub = X_train[sub_idx], y_train[sub_idx], g_train[sub_idx]")

ZSCORE_CELL = '''# ---- A4: per-subject z-score of the feature windows (same idea as the 1D-CNN in notebook 06) ----
PER_SUBJECT_ZSCORE = False   # False = old tables (only A5 changed). True = per-subject z-scored features.

def zscore_per_subject(df, feature_cols, subject_col='subject_id'):
    """Z-score each feature within each subject (all of that subject's windows). Label-free."""
    out = df.copy()
    out[feature_cols] = (out.groupby(subject_col)[feature_cols]
                         .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8)))
    return out

if PER_SUBJECT_ZSCORE:
    df_wesad_bin = zscore_per_subject(df_wesad_bin, wesad_feat_cols)
    df_wesad_3cl = zscore_per_subject(df_wesad_3cl, wesad_feat_cols)
print('Per-subject z-score:', PER_SUBJECT_ZSCORE)
'''

def sub_call(t, label):
    t2, n = SUB_RE.subn(sub_call_repl, t)
    if n == 1:
        return t2
    if "stratified_subsample_idx(" in t:
        print(f"  [skip] {label}: already applied")
        return t
    raise SystemExit(f"\nSTOPPED (nothing written): [{label}] could not find the stratified_subsample(...) call. "
                     "Send this message to Claude.")

def add_zscore_cell(nb, loader_needle):
    if any("PER_SUBJECT_ZSCORE" in src(c) for c in nb["cells"]):
        print("  [skip] z-score cell: already present")
        return
    load_cell = find_cell(nb, loader_needle)
    pos = next(i for i, c in enumerate(nb["cells"]) if c is load_cell)
    nb["cells"].insert(pos + 1, new_code_cell(nb, ZSCORE_CELL))

# ---------------------------------------------------------------- notebook 05
def patch_05():
    nb = load("05_classical_baselines.ipynb")
    c = find_cell(nb, "INNER_CV = Stratified")
    t = src(c)
    t = sub1(t, IMPORT_OLD, IMPORT_NEW, "05 import", done="GridSearchCV, StratifiedGroupKFold")
    t = sub1(t, CV_OLD, CV_NEW, "05 INNER_CV", done="INNER_CV = StratifiedGroupKFold(")
    t = sub1(t, PRINT_OLD, PRINT_NEW, "05 config print", soft=True)
    setsrc(c, t)

    c = find_cell(nb, "def stratified_subsample(")
    t = src(c)
    if "def stratified_subsample_idx" in t:
        print("  [skip] 05 idx fn: already applied")
    else:
        t = sub1(t, "def compute_metrics(", IDX_FN + "def compute_metrics(", "05 idx fn")
    t = sub_call(t, "05 subsample call")
    t = sub1(t, "search.fit(X_train_sub, y_train_sub)",
                "search.fit(X_train_sub, y_train_sub, groups=g_train_sub)", "05 fit", done="groups=g_train_sub")
    t = sub1(t, "Inner CV        : StratifiedKFold(5)",
                "Inner CV        : StratifiedGroupKFold(5), grouped by subject", "05 run print", soft=True)
    t = sub1(t, "inner 5-fold StratifiedKFold on the training subsample",
                "inner 5-fold subject-grouped StratifiedGroupKFold on the training subsample", "05 doc", soft=True)
    setsrc(c, t)

    c = find_cell(nb, "def run_loso_svm_pca_matched")
    t = src(c)
    t = sub_call(t, "05 pca subsample call")
    t = sub1(t, "search.fit(X_train_sub, y_train_sub)",
                "search.fit(X_train_sub, y_train_sub, groups=g_train_sub)", "05 pca fit", done="groups=g_train_sub")
    setsrc(c, t)

    add_zscore_cell(nb, "wesad_features_binary.csv")
    return nb

# ---------------------------------------------------------------- notebook 05b
def patch_05b():
    nb = load("05b_classical_baselines_full.ipynb")
    c = find_cell(nb, "INNER_CV = Stratified")
    t = src(c)
    t = sub1(t, IMPORT_OLD, IMPORT_NEW, "05b import", done="GridSearchCV, StratifiedGroupKFold")
    t = sub1(t, CV_OLD, CV_NEW, "05b INNER_CV", done="INNER_CV = StratifiedGroupKFold(")
    t = sub1(t, PRINT_OLD, PRINT_NEW, "05b config print", soft=True)
    setsrc(c, t)

    c = find_cell(nb, "def run_loso_full")
    t = src(c)
    if "g_train = df.loc[train_mask, subject_col]" in t:
        print("  [skip] 05b g_train: already applied")
    else:
        t = sub1(t, "        y_test  = df.loc[test_mask,  label_col].values\n",
                    "        y_test  = df.loc[test_mask,  label_col].values\n"
                    "        g_train = df.loc[train_mask, subject_col].values   # subject id of each training window\n",
                    "05b g_train")
    t = sub1(t, "search.fit(X_train, y_train)", "search.fit(X_train, y_train, groups=g_train)", "05b fit", done="groups=g_train")
    t = sub1(t, "Inner CV : StratifiedKFold(5)", "Inner CV : StratifiedGroupKFold(5), grouped by subject", "05b run print", soft=True)
    setsrc(c, t)

    c = nb["cells"][0]
    setsrc(c, sub1(src(c), PRINT_OLD, PRINT_NEW, "05b markdown", soft=True))

    add_zscore_cell(nb, "wesad_features_binary.csv")
    return nb

# ---------------------------------------------------------------- notebook 09 markdown
N09 = "09_hybrid_cnn_qnn.ipynb" if (NB / "09_hybrid_cnn_qnn.ipynb").exists() else "09_hybrid_mlp_qnn.ipynb"

def patch_09():
    nb = load(N09)
    c = find_cell(nb, "Why Hand-Crafted Features", "markdown")
    t = src(c)
    t = sub1(t,
        "The standalone 1D-CNN on raw signals failed under LOSO (F1=0.39) due to insufficient\n"
        "data for convolutional filter learning. Hand-crafted features provide a stable,\n"
        "physiologically meaningful input representation that allows the classical front-end\n"
        "to learn a compact encoding for the quantum layer.",
        "Hand-crafted features give a compact, physiologically meaningful input for a small MLP\n"
        "front-end that encodes them into qubit angles for the quantum layer. The 1D-CNN in\n"
        "notebook 06 is a separate raw-signal baseline.", "09 stale paragraph")
    t = sub1(t, "  -> MinMaxScaler(-pi, pi)     # angle encoding",
                "  -> Tanh, then x pi           # angle encoding in (-pi, pi)", "09 arch text")
    t = sub1(t, "- RobustScaler + MinMaxScaler(-pi, pi) preprocessing",
                "- RobustScaler preprocessing (training fold); angle encoding = Tanh x pi inside the model",
                "09 data conditions")
    setsrc(c, t)
    return nb

# ---------------------------------------------------------------- rename
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout

def rename():
    for old in ["notebooks/09_hybrid_cnn_qnn.ipynb",
                "results/old/hybrid_cnn_qnn_per_fold.csv", "results/old/hybrid_cnn_qnn_summary.csv",
                "results/output_data/hybrid_cnn_qnn_per_fold.csv", "results/output_data/hybrid_cnn_qnn_summary.csv",
                "results/plots/09_hybrid_cnn_qnn"]:
        if pathlib.Path(old).exists():
            git("mv", old, old.replace("cnn_qnn", "mlp_qnn"))
    SUBS = [("CNN-QNN", "MLP-QNN"), ("CNNQNN", "MLPQNN"), ("cnn_qnn", "mlp_qnn")]
    for f in git("ls-files").splitlines():
        if f.endswith((".ipynb", ".md", ".csv", ".txt", ".json")) and f != "apply_week1.py":
            p = pathlib.Path(f); s = p.read_text(encoding="utf-8"); t = s
            for a, b in SUBS: t = t.replace(a, b)
            if t != s:
                p.write_text(t, encoding="utf-8", newline=""); print("  renamed text in", f)

if __name__ == "__main__":
    if not (NB / "05_classical_baselines.ipynb").exists():
        sys.exit("Run this from the repo root (the folder that contains notebooks/).")
    new05, new05b, new09 = patch_05(), patch_05b(), patch_09()      # all asserted BEFORE writing
    save("05_classical_baselines.ipynb", new05)
    save("05b_classical_baselines_full.ipynb", new05b)
    save(N09, new09)
    print("A5 + A4-cell + notebook-09 markdown patched. Now renaming CNN-QNN -> MLP-QNN ...")
    rename()
    print("Done. Next: open 05 and 05b, Restart & Run All.")