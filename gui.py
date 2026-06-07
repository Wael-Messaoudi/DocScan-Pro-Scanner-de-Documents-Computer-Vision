"""
gui.py — Interface graphique DocScan Pro (Tkinter, thème sombre)
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2, threading, os
from scanner import (
    apply_gaussian, apply_median, apply_bilateral,
    apply_canny, apply_sobel,
    apply_threshold_global, apply_threshold_adaptive, apply_threshold_otsu,
    apply_roi_mask, apply_erosion, apply_dilation, apply_opening, apply_closing,
    apply_harris_corners, apply_blob_detection, scan_document,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG, SURFACE, CARD, BORDER          = "#0d1117","#161b22","#1c2128","#30363d"
ACCENT, ACCENT2, SUCCESS           = "#58a6ff","#bc8cff","#3fb950"
TEXT, DIM                          = "#e6edf3","#8b949e"
FH = ("Georgia",20,"bold");  FB = ("Helvetica",10)
FS = ("Helvetica",9);        FM = ("Courier",9);  FBT = ("Helvetica",10,"bold")

# Valeurs par défaut (remplacent les sliders)
GK, MK, CLO, CHI, TV, MK2 = 5, 5, 50, 150, 127, 5


# ═══════════════════════════════════════════════════════
#  WIDGETS RÉUTILISABLES
# ═══════════════════════════════════════════════════════

class ImagePanel(tk.Frame):
    """Cadre sombre affichant une image PIL, redimensionnée automatiquement."""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD, highlightbackground=BORDER,
                         highlightthickness=1, **kw)
        self.lbl = tk.Label(self, bg=CARD, text="—", fg=DIM, font=FS)
        self.lbl.pack(expand=True, fill="both", padx=6, pady=6)
        self._img = None

    def show(self, pil, mw=500, mh=460):
        w, h = pil.size; r = min(mw/w, mh/h)
        pil = pil.resize((max(1,int(w*r)), max(1,int(h*r))), Image.LANCZOS)
        self._img = ImageTk.PhotoImage(pil)
        self.lbl.config(image=self._img, text="")

    def clear(self):
        self._img = None; self.lbl.config(image="", text="—")


class FlatBtn(tk.Button):
    """Bouton plat avec effet hover (éclaircissement au survol)."""
    def __init__(self, parent, text, color, cmd=None, state="normal", **kw):
        super().__init__(parent, text=text, font=FBT, bg=color, fg=TEXT,
                         activebackground=color, relief="flat", cursor="hand2",
                         padx=14, pady=7, state=state, command=cmd, **kw)
        # Calcule une couleur légèrement plus claire pour l'effet hover
        def lig(c): r,g,b=int(c[1:3],16),int(c[3:5],16),int(c[5:7],16); return f"#{min(r+25,255):02x}{min(g+25,255):02x}{min(b+25,255):02x}"
        self.bind("<Enter>", lambda e: self.config(bg=lig(color)))
        self.bind("<Leave>", lambda e: self.config(bg=color))


# ═══════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ═══════════════════════════════════════════════════════

class DocScanApp:

    def __init__(self, root):
        self.root = root
        self.root.title("DocScan Pro — Vision par Ordinateur")
        self.root.configure(bg=BG); self.root.geometry("1200x780")
        # État interne : image chargée et résultats
        self.img_path = self.cv_img = self.res_pil = self.scan_pil = None
        self._styles(); self._header(); self._main(); self._footer()

    # ── Styles ttk ────────────────────────────────────────────────────────────
    def _styles(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=SURFACE, foreground=DIM, padding=[14,8], font=FB)
        s.map("TNotebook.Tab", background=[("selected",CARD)], foreground=[("selected",ACCENT)])
        s.configure("TProgressbar", troughcolor=SURFACE, background=ACCENT, bordercolor=BORDER)

    # ── En-tête ───────────────────────────────────────────────────────────────
    def _header(self):
        h = tk.Frame(self.root, bg=SURFACE, pady=14); h.pack(fill="x")
        tk.Label(h, text="📷  DocScan Pro", font=FH, fg=TEXT, bg=SURFACE).pack(side="left", padx=24)
        tk.Label(h, text="Scanner de documents — Vision par Ordinateur classique",
                 font=FS, fg=DIM, bg=SURFACE).pack(side="left")
        # Boutons globaux (droite)
        bf = tk.Frame(h, bg=SURFACE); bf.pack(side="right", padx=16)
        FlatBtn(bf, "📂 Charger image", ACCENT, self._load).pack(side="left", padx=4)
        self.btn_save = FlatBtn(bf, "💾 Sauvegarder", SUCCESS, self._save, state="disabled")
        self.btn_save.pack(side="left", padx=4)
        self.btn_pdf = FlatBtn(bf, "📄 Exporter PDF", ACCENT2, self._pdf, state="disabled")
        self.btn_pdf.pack(side="left", padx=4)
        tk.Frame(self.root, height=1, bg=BORDER).pack(fill="x")

    # ── Corps (panneau gauche + droite) ───────────────────────────────────────
    def _main(self):
        p = tk.PanedWindow(self.root, orient="horizontal", bg=BG, sashwidth=4)
        p.pack(fill="both", expand=True, padx=8, pady=8)
        left = tk.Frame(p, bg=SURFACE, width=250); p.add(left, minsize=210)
        right = tk.Frame(p, bg=BG);                p.add(right, minsize=500)
        self._controls(left); self._viewer(right)

    # ── Panneau de contrôles (scrollable) ─────────────────────────────────────
    def _controls(self, parent):
        # Canvas scrollable pour la liste de boutons
        cv = tk.Canvas(parent, bg=SURFACE, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); cv.pack(side="left", fill="both", expand=True)
        inn = tk.Frame(cv, bg=SURFACE)
        cv.create_window((0,0), window=inn, anchor="nw")
        inn.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

        # Helpers pour créer sections et boutons rapidement
        def sec(t, c=ACCENT):
            tk.Label(inn, text=f"  {t}", font=("Helvetica",10,"bold"),
                     fg=c, bg=SURFACE, anchor="w").pack(fill="x", pady=(14,2))
            tk.Frame(inn, height=1, bg=c).pack(fill="x", padx=8)
        def btn(t, cmd): FlatBtn(inn, t, CARD, cmd).pack(fill="x", padx=12, pady=2)

        # ── Scanner complet ──────────────────────────────────────────────────
        sec("🔬  Scanner de documents")
        btn("▶  Lancer le scanner complet", self._scan)

        # ── Filtres ──────────────────────────────────────────────────────────
        sec("🌫️  Filtres")
        btn("Appliquer Gaussien",  lambda: self._ap("Filtre Gaussien",  lambda i: apply_gaussian(i, GK)))
        btn("Appliquer Médian",    lambda: self._ap("Filtre Médian",    lambda i: apply_median(i, MK)))
        btn("Appliquer Bilatéral", lambda: self._ap("Filtre Bilatéral", apply_bilateral))

        # ── Contours ─────────────────────────────────────────────────────────
        sec("🔍  Détection de contours")
        btn("Canny", lambda: self._ap("Canny", lambda i: apply_canny(i, CLO, CHI)))
        btn("Sobel", lambda: self._ap("Sobel", apply_sobel))

        # ── Seuillage ────────────────────────────────────────────────────────
        sec("⬛  Seuillage")
        btn("Seuillage global",    lambda: self._ap("Seuillage Global",    lambda i: apply_threshold_global(i, TV)))
        btn("Seuillage adaptatif", lambda: self._ap("Seuillage Adaptatif", apply_threshold_adaptive))
        btn("Otsu",                lambda: self._ap("Seuillage Otsu",      apply_threshold_otsu))

        # ── ROI ──────────────────────────────────────────────────────────────
        sec("🟦  Masque / ROI")
        btn("Appliquer ROI (centre 80%)", lambda: self._ap("Masque ROI", apply_roi_mask))

        # ── Morphologie ──────────────────────────────────────────────────────
        sec("🔷  Morphologie")
        for label, fn in [("Érosion",    lambda i: apply_erosion(i, MK2)),
                          ("Dilatation", lambda i: apply_dilation(i, MK2)),
                          ("Ouverture",  lambda i: apply_opening(i, MK2)),
                          ("Fermeture",  lambda i: apply_closing(i, MK2))]:
            btn(label, lambda fn=fn, l=label: self._ap(l, fn))

        # ── Caractéristiques ─────────────────────────────────────────────────
        sec("📍  Caractéristiques")
        btn("Coins Harris", lambda: self._feat("Coins Harris", apply_harris_corners))
        btn("Blobs",        lambda: self._feat("Blobs",        apply_blob_detection))


    # ── Visualiseur (3 onglets) ───────────────────────────────────────────────
    def _viewer(self, parent):
        self.nb = ttk.Notebook(parent); self.nb.pack(fill="both", expand=True)

        # Onglet 1 : Scanner (3 colonnes : original / contour / résultat)
        ts = tk.Frame(self.nb, bg=BG); self.nb.add(ts, text="  🔬 Scanner  ")
        for c in range(3): ts.columnconfigure(c, weight=1)
        ts.rowconfigure(1, weight=1)
        for c, t in enumerate(["📷 Original", "🔲 Contour détecté", "✨ Résultat scanné"]):
            tk.Label(ts, text=t, font=FS, fg=DIM, bg=BG).grid(row=0, column=c, pady=4)
        self.p_orig = ImagePanel(ts); self.p_cont = ImagePanel(ts); self.p_scan = ImagePanel(ts)
        self.p_orig.grid(row=1,column=0,sticky="nsew",padx=4,pady=4)
        self.p_cont.grid(row=1,column=1,sticky="nsew",padx=4,pady=4)
        self.p_scan.grid(row=1,column=2,sticky="nsew",padx=4,pady=4)

        # Onglet 2 : Traitements (2 colonnes : source / résultat)
        tp = tk.Frame(self.nb, bg=BG); self.nb.add(tp, text="  ⚙️ Traitements  ")
        for c in range(2): tp.columnconfigure(c, weight=1)
        tp.rowconfigure(1, weight=1)
        tk.Label(tp, text="📷 Image source", font=FS, fg=DIM, bg=BG).grid(row=0,column=0,pady=4)
        tk.Label(tp, text="⚙️ Résultat",    font=FS, fg=DIM, bg=BG).grid(row=0,column=1,pady=4)
        self.p_src = ImagePanel(tp); self.p_res = ImagePanel(tp)
        self.p_src.grid(row=1,column=0,sticky="nsew",padx=4,pady=4)
        self.p_res.grid(row=1,column=1,sticky="nsew",padx=4,pady=4)

        # Onglet 3 : Log d'analyse
        ta = tk.Frame(self.nb, bg=BG); self.nb.add(ta, text="  📊 Analyse  ")
        self.log = tk.Text(ta, bg=CARD, fg=TEXT, font=FM, relief="flat",
                           padx=12, pady=12, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    # ── Pied de page ──────────────────────────────────────────────────────────
    def _footer(self):
        tk.Frame(self.root, height=1, bg=BORDER).pack(fill="x", side="bottom")
        f = tk.Frame(self.root, bg=SURFACE, pady=6); f.pack(fill="x", side="bottom")
        self.prog = ttk.Progressbar(f, orient="horizontal", mode="determinate", maximum=8, length=280)
        self.prog.pack(side="left", padx=16)
        self.lbl_st = tk.Label(f, text="Prêt — chargez une image.", font=FS, fg=DIM, bg=SURFACE)
        self.lbl_st.pack(side="left", padx=8)
        self.lbl_t = tk.Label(f, text="", font=FS, fg=ACCENT, bg=SURFACE)
        self.lbl_t.pack(side="right", padx=16)

    # ── Helpers internes ──────────────────────────────────────────────────────

    def _status(self, msg, p=None):
        """Met à jour la barre de statut et éventuellement la progression."""
        self.lbl_st.config(text=msg)
        if p is not None: self.prog["value"] = p
        self.root.update_idletasks()

    def _log(self, txt):
        """Ajoute une ligne dans le panneau Analyse."""
        self.log.config(state="normal"); self.log.insert("end", txt+"\n")
        self.log.see("end"); self.log.config(state="disabled")

    def _pil(self, cv):
        """Convertit une image OpenCV (BGR ou Gray) en PIL RGB."""
        return Image.fromarray(cv) if cv.ndim==2 else Image.fromarray(cv2.cvtColor(cv,cv2.COLOR_BGR2RGB))

    def _check(self):
        """Vérifie qu'une image est chargée, affiche un avertissement sinon."""
        if self.cv_img is None:
            messagebox.showwarning("Attention","Chargez d'abord une image."); return False
        return True

    # ── Chargement d'image ────────────────────────────────────────────────────
    def _load(self):
        path = filedialog.askopenfilename(title="Choisir une image",
               filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff"),("Tous","*.*")])
        if not path: return
        img = cv2.imread(path)
        if img is None: messagebox.showerror("Erreur","Impossible de lire l'image."); return
        self.img_path = path; self.cv_img = img; pil = self._pil(img)
        self.p_orig.show(pil,350,460); self.p_src.show(pil,500,460)
        for p in (self.p_cont, self.p_scan, self.p_res): p.clear()
        self.res_pil = self.scan_pil = None
        self.btn_save.config(state="disabled"); self.btn_pdf.config(state="disabled")
        self._status(f"Image chargée : {os.path.basename(path)}", 0)
        self.log.config(state="normal"); self.log.delete("1.0","end"); self.log.config(state="disabled")
        self._log(f"📂 {path}  |  {img.shape[1]}×{img.shape[0]} px")

    # ── Scanner complet (thread) ──────────────────────────────────────────────
    def _scan(self):
        if not self._check(): return
        self.nb.select(0); self._status("Démarrage du scanner…", 0)
        threading.Thread(target=self._scan_th, daemon=True).start()

    def _scan_th(self):
        """Thread : lance le pipeline scanner et met à jour l'UI à la fin."""
        def cb(s, _, m): self.root.after(0, lambda s=s,m=m: self._status(m, s))
        orig, scanned, cont, err = scan_document(self.img_path, cb)
        if err: self.root.after(0, lambda: messagebox.showerror("Erreur", err)); return
        op, cp, sp = self._pil(orig), self._pil(cont), Image.fromarray(scanned)
        self.scan_pil = self.res_pil = sp
        def up():
            self.p_orig.show(op,350,460); self.p_cont.show(cp,350,460); self.p_scan.show(sp,350,460)
            self.btn_save.config(state="normal"); self.btn_pdf.config(state="normal")
            self._log("\n🔬 Scanner OK — Gaussien→Canny→Morphologie→Perspective→Seuillage adaptatif")
        self.root.after(0, up)

    # ── Traitements individuels ───────────────────────────────────────────────

    def _ap(self, name, fn):
        """Applique un traitement (img→img) et affiche le résultat."""
        if not self._check(): return
        self.nb.select(1); result, ms = fn(self.cv_img.copy())
        pil = self._pil(result); self.res_pil = pil
        self.p_res.show(pil, 500, 460); self.btn_save.config(state="normal")
        self._status(f"✅ {name}", 1); self.lbl_t.config(text=f"⏱ {ms} ms")
        self._log(f"\n⚙️ {name}  |  {ms} ms  |  {result.shape[1]}×{result.shape[0]}")

    def _feat(self, name, fn):
        """Applique une détection de caractéristiques (img→img, ms, nb)."""
        if not self._check(): return
        self.nb.select(1); result, ms, n = fn(self.cv_img.copy())
        pil = self._pil(result); self.res_pil = pil
        self.p_res.show(pil, 500, 460); self.btn_save.config(state="normal")
        self._status(f"✅ {name} — {n} détection(s)"); self.lbl_t.config(text=f"⏱ {ms} ms")
        self._log(f"\n📍 {name}  |  {n} détections  |  {ms} ms")

    def _seg(self, name, fn):
        """Applique une segmentation (img→img, ms, nb_segments)."""
        if not self._check(): return
        result, ms, nb = fn(self.cv_img.copy())
        pil = self._pil(result); self.res_pil = pil
        self.p_res.show(pil, 500, 460); self.btn_save.config(state="normal")
        self.nb.select(2); self._status(f"✅ {name} — {nb} segment(s)")
        self.lbl_t.config(text=f"⏱ {ms} ms")
        self._log(f"\n🗺️ {name}  |  {nb} segments  |  {ms} ms")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    def _save(self):
        if not self.res_pil: return
        p = filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("Tous","*.*")])
        if not p: return
        self.res_pil.save(p); messagebox.showinfo("✅","Sauvegardé :\n"+p)
        self._log(f"\n💾 {p}")

    def _pdf(self):
        img = self.scan_pil or self.res_pil
        if not img: return
        p = filedialog.asksaveasfilename(defaultextension=".pdf",
            filetypes=[("PDF","*.pdf")])
        if not p: return
        try:
            img.convert("RGB").save(p,"PDF",resolution=200)
            messagebox.showinfo("✅","PDF sauvegardé :\n"+p); self._log(f"\n📄 {p}")
        except Exception as e:
            messagebox.showerror("Erreur PDF", str(e))


# ── Point d'entrée ────────────────────────────────────────────────────────────
def run_app():
    root = tk.Tk(); DocScanApp(root); root.mainloop()