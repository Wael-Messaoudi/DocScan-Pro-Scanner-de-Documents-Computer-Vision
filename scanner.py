"""
scanner.py — Techniques classiques de Vision par Ordinateur (OpenCV)
Filtres · Contours · Seuillage · ROI · Morphologie · Caractéristiques · Segmentation · Scanner
"""

"""pipeline : Chargement
    ↓
Redimensionnement (resize_keep_ratio)
    ↓
Filtre Gaussien (réduction bruit)
    ↓
Détection Canny (contours)
    ↓
Morphologie : Dilatation → Érosion (nettoyer les bords)
    ↓
Recherche contour quadrilatère du document
    ↓
Correction de perspective (four_point_transform)
    ↓
Seuillage adaptatif (effet scanner noir & blanc)"""

import cv2, numpy as np, time

# ═══════════════════════════════════════════════════════
#  UTILITAIRES INTERNES
# ═══════════════════════════════════════════════════════

# Convertit en niveaux de gris si l'image est en couleur
def _gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

# Convertit en BGR si l'image est en niveaux de gris
def _bgr(img):  return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

# Retourne le temps écoulé depuis t0 en millisecondes
def _ms(t0):    return round((time.time() - t0) * 1000, 1)

# Crée un noyau carré de k×k (pour la morphologie)
def _ker(k):    return np.ones((k, k), np.uint8)

def _binary(img, k=5):
    """Gris → flou gaussien → seuillage Otsu inversé (préparation morphologie)."""
    _, b = cv2.threshold(cv2.GaussianBlur(_gray(img), (k,k), 0),
                         0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b

def resize_keep_ratio(img, max_h=800):
    """Redimensionne en conservant le ratio largeur/hauteur."""
    h, w = img.shape[:2]
    return cv2.resize(img, (int(max_h * w / h), max_h)), w / int(max_h * w / h)

def order_points(pts):
    """Ordonne 4 points : haut-gauche, haut-droite, bas-droite, bas-gauche."""
    s, d = pts.sum(1), np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")

def four_point_transform(img, pts):
    """Correction de perspective à partir de 4 points (redresse le document)."""
    tl, tr, br, bl = order_points(pts)
    W = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    H = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(order_points(pts), dst), (W, H))

# ═══════════════════════════════════════════════════════
#  1. FILTRES
# ═══════════════════════════════════════════════════════

def apply_gaussian(img, k=5):
    """Flou gaussien — réduit le bruit (ksize impair obligatoire)."""
    t0 = time.time(); k = k if k % 2 else k + 1
    return cv2.GaussianBlur(img, (k,k), 0), _ms(t0)

def apply_median(img, k=5):
    """Flou médian — efficace contre le bruit sel & poivre.Le filtre médian supprime surtout :
bruit sel et poivre
pixels noirs/blancs isolés"""
    t0 = time.time(); k = k if k % 2 else k + 1
    return cv2.medianBlur(img, k), _ms(t0)

def apply_bilateral(img):
    """Flou bilatéral — lisse tout en préservant les contours.réduit le bruit
lisse les zones uniformes
conserve fortement les contours"""
    t0 = time.time()
    return cv2.bilateralFilter(img, 9, 75, 75), _ms(t0)

# ═══════════════════════════════════════════════════════
#  2. DÉTECTION DE CONTOURS
# ═══════════════════════════════════════════════════════

def apply_canny(img, lo=50, hi=150):
    """Canny — double seuil (lo/hi) pour détecter les contours nets.détecter les contours importants d’une image
identifier les formes et objets
extraire les bords nets"""
    t0 = time.time()
    return _bgr(cv2.Canny(cv2.GaussianBlur(_gray(img),(5,5),0), lo, hi)), _ms(t0)

def apply_sobel(img):
    """Sobel — gradients Gx + Gy combinés en magnitude totale.détecter les bords
mesurer les gradients
repérer les changements brusques de pixels"""
    t0 = time.time(); g = cv2.GaussianBlur(_gray(img), (3,3), 0)
    mag = cv2.magnitude(cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3),
                        cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3))
    return _bgr(cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)), _ms(t0)

# ═══════════════════════════════════════════════════════
#  3. SEUILLAGE
# ═══════════════════════════════════════════════════════

def apply_threshold_global(img, v=127):
    """Seuillage global : pixel > v → blanc, sinon → noir."""
    t0 = time.time()
    _, r = cv2.threshold(_gray(img), v, 255, cv2.THRESH_BINARY)
    return _bgr(r), _ms(t0)

def apply_threshold_adaptive(img):
    """Seuillage adaptatif gaussien — robuste aux variations d'éclairage."""
    t0 = time.time()
    r = cv2.adaptiveThreshold(_gray(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 11, 2)
    return _bgr(r), _ms(t0)

def apply_threshold_otsu(img):
    """Seuillage Otsu — trouve automatiquement le seuil optimal (histogramme bimodal)."""
    t0 = time.time()
    _, r = cv2.threshold(cv2.GaussianBlur(_gray(img),(5,5),0),
                         0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _bgr(r), _ms(t0)

# ═══════════════════════════════════════════════════════
#  4. MASQUE / ROI
# ═══════════════════════════════════════════════════════

def apply_roi_mask(img):
    """Masque ROI rectangulaire centré (80% de l'image) — isole la zone d'intérêt.
    garde uniquement la région centrale"""
    t0 = time.time(); h, w = img.shape[:2]
    # Points du rectangle (10% de marge de chaque côté)
    p1, p2 = (int(w*.1), int(h*.1)), (int(w*.9), int(h*.9))
    #Création du masque noir | Remplissage de la ROI en blanc
    mask = np.zeros((h,w), np.uint8); mask[p1[1]:p2[1], p1[0]:p2[0]] = 255
    m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else mask
    r = _bgr(cv2.bitwise_and(img, m3))
    cv2.rectangle(r, p1, p2, (0,200,255), 2)   # contour jaune pour visualiser
    return r, _ms(t0)

# ═══════════════════════════════════════════════════════
#  5. MORPHOLOGIE MATHÉMATIQUE
# ═══════════════════════════════════════════════════════

def apply_erosion(img, k=5):
    """Érosion — réduit les objets blancs, supprime le bruit."""
    t0 = time.time(); return _bgr(cv2.erode(_binary(img), _ker(k))), _ms(t0)

def apply_dilation(img, k=5):
    """Dilatation — agrandit les objets blancs, comble les trous."""
    t0 = time.time(); return _bgr(cv2.dilate(_binary(img), _ker(k))), _ms(t0)

def apply_opening(img, k=5):
    """Ouverture (érosion→dilatation) — supprime le bruit fin.
    bruit supprimé
 objets principaux conservés"""
    t0 = time.time()
    return _bgr(cv2.morphologyEx(_binary(img), cv2.MORPH_OPEN, _ker(k))), _ms(t0)

def apply_closing(img, k=5):
    """Fermeture (dilatation→érosion) — bouche les trous dans les objets.
    trous remplis
    formes plus compactes
    contours plus continus"""
    t0 = time.time()
    return _bgr(cv2.morphologyEx(_binary(img), cv2.MORPH_CLOSE, _ker(k))), _ms(t0)

# ═══════════════════════════════════════════════════════
#  6. CARACTÉRISTIQUES
# ═══════════════════════════════════════════════════════

def apply_harris_corners(img):
    """Coins Harris — détecte les intersections de bords (marqués en rouge)."""
    t0 = time.time()
    dst = cv2.dilate(cv2.cornerHarris(np.float32(_gray(img)), 2, 3, 0.04), None)
    r = _bgr(img.copy()); r[dst > 0.01 * dst.max()] = [0, 0, 255]
    return r, _ms(t0), int(np.sum(dst > 0.01 * dst.max()))

def apply_blob_detection(img):
    """Blobs — repère les régions homogènes par aire, circularité et convexité."""
    t0 = time.time(); p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True;        p.minArea = 200;  p.maxArea = 50000
    #Mesure à quel point un objet ressemble à :un cercle.
    p.filterByCircularity = True; p.minCircularity = 0.3
    #si la forme possède des creux.
    p.filterByConvexity   = True; p.minConvexity   = 0.5
    kps = cv2.SimpleBlobDetector_create(p).detect(_gray(img))
    r = cv2.drawKeypoints(_bgr(img), kps, np.array([]), (0,255,0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return r, _ms(t0), len(kps)


# ═══════════════════════════════════════════════════════
#  7. PIPELINE SCANNER DE DOCUMENTS
# ═══════════════════════════════════════════════════════

def _find_doc(edged):
    """Cherche le plus grand contour quadrilatère dans une image de bords."""
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4: return approx
    return None

def scan_document(image_path, callback=None):
    """
    Pipeline complet :
    Chargement → Resize → Gaussien → Canny + Morphologie
    → Contour doc → Perspective → Seuillage adaptatif (effet scanner)

    Retourne : (img_originale, img_scannée, img_contour, erreur|None)
    """
    def cb(n, msg):
        if callback: callback(n, 8, msg)

    cb(1, "Chargement…")
    img = cv2.imread(image_path)
    if img is None: return None, None, None, "❌ Impossible de lire l'image."

    cb(2, "Redimensionnement…")
    small, scale = resize_keep_ratio(img, 800)

    cb(3, "Filtre gaussien…")
    blur = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5,5), 0)

    cb(4, "Canny + Morphologie (dilatation→érosion)…")
    k = _ker(5)
    edges = cv2.erode(cv2.dilate(cv2.Canny(blur, 50, 150), k, iterations=2), k)

    cb(5, "Recherche du document…")
    doc = _find_doc(edges)
    if doc is None:   # fallback : seuillage adaptatif
        th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
        doc = _find_doc(cv2.erode(cv2.dilate(th, k, iterations=2), k))

    vis = small.copy()
    if doc is None:   # fallback final : image entière
        h, w = small.shape[:2]
        doc = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]).reshape(4,1,2)
        cb(5, "⚠️ Document non détecté — image complète utilisée")
    else:
        cv2.drawContours(vis, [doc], -1, (0,255,0), 3)

    cb(6, "Correction de perspective…")
    warped = four_point_transform(img, doc.reshape(4,2).astype("float32") * scale)

    cb(7, "Effet scanner (seuillage adaptatif)…")
    final = cv2.adaptiveThreshold(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY),
                                  255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    cb(8, "✅ Terminé !")
    return small, final, vis, None