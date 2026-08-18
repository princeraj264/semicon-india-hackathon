#!/usr/bin/env python3
"""Generate the KLA PS1 Unified Blind Restoration Network architecture PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table,
    TableStyle, PageBreak, Flowable,
)

# ---------- palette ----------
INK = HexColor("#1a1f2e")
ACCENT = HexColor("#0f4c81")
ACCENT_LT = HexColor("#e8f0f8")
GRAY = HexColor("#5a6472")
LINE = HexColor("#c9d2dd")
WARN = HexColor("#8a3b12")
OK = HexColor("#1e6b3a")
BOX_FILL = HexColor("#f3f6fa")

W, H = A4

# ---------- styles ----------
def st(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK)
    base.update(kw)
    return ParagraphStyle(name, **base)

S_TITLE = st("title", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK)
S_SUB = st("sub", fontSize=11, leading=15, textColor=GRAY)
S_H1 = st("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=ACCENT, spaceBefore=14, spaceAfter=5)
S_H2 = st("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=9, spaceAfter=3)
S_BODY = st("body", spaceAfter=5)
S_BULL = st("bull", leftIndent=12, bulletIndent=2, spaceAfter=3)
S_CELL = st("cell", fontSize=8.5, leading=11.5)
S_CELLB = st("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11.5)
S_CELLH = st("cellh", fontName="Helvetica-Bold", fontSize=8.5, leading=11.5, textColor=HexColor("#ffffff"))
S_CODE = st("code", fontName="Courier", fontSize=8.2, leading=11, textColor=INK, backColor=BOX_FILL, borderPadding=6, spaceAfter=6)
S_CAP = st("cap", fontSize=8, leading=10.5, textColor=GRAY, alignment=TA_CENTER, spaceAfter=8)

def P(text, style=S_BODY):
    return Paragraph(text, style)

def bullet(text):
    return Paragraph(f"&bull;&nbsp;&nbsp;{text}", S_BULL)

# ---------- pipeline diagram flowable ----------
class Pipeline(Flowable):
    def __init__(self, width=170 * mm):
        super().__init__()
        self.width = width
        self.rows = [
            ("INPUT", "Degraded .npy  (raw float32, UNCLIPPED - speckle overshoot kept as signal)", "KLA rule: do not clip inputs"),
            ("STAGE 0", "Noise-Level Estimator - tiny 3-layer CNN produces per-pixel noise map M; concat as 2nd channel", "FFDNet (Zhang et al., 2018)"),
            ("STAGE 1", "Shallow Feature Extraction - single 3x3 conv, C=48", "SwinIR (Liang et al., 2021)"),
            ("STAGE 2", "Deep Body - 4-level U-shaped encoder-decoder of MDTA + GDFN transformer blocks\n[4, 6, 6, 8] blocks; channel attention (CxC), linear in resolution; ~5-15M params", "Restormer (Zamir et al., 2022)"),
            ("SKIP", "Long global skip: shallow features added back before reconstruction (low-freq bypass)", "SwinIR Eq.5 / DnCNN residual"),
            ("STAGE 3", "Reconstruction Head - conv + PixelShuffle x2 or x4 (scale inferred at runtime from input shape)", "SwinIR / W2S-ESRGAN"),
            ("OUTPUT", "nan_to_num  ->  clip[0,1]  ->  squeeze to (H,W) float32  ->  save same filename .npy", "KLA output contract"),
        ]
        self.rh = 13.5 * mm
        self.gap = 4.5 * mm
        self.height = len(self.rows) * self.rh + (len(self.rows) - 1) * self.gap + 4 * mm

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        x0 = 2 * mm
        bw = self.width - 46 * mm
        y = self.height - self.rh - 2 * mm
        for i, (tag, body, cite) in enumerate(self.rows):
            c.setFillColor(BOX_FILL)
            c.setStrokeColor(ACCENT if tag in ("INPUT", "OUTPUT") else LINE)
            c.setLineWidth(1.1 if tag in ("INPUT", "OUTPUT") else 0.8)
            c.roundRect(x0, y, bw, self.rh, 2.5 * mm, stroke=1, fill=1)
            c.setFillColor(ACCENT)
            c.roundRect(x0 + 2.5 * mm, y + self.rh - 6.2 * mm, 22 * mm, 4.6 * mm, 1.5 * mm, stroke=0, fill=1)
            c.setFillColor(HexColor("#ffffff"))
            c.setFont("Helvetica-Bold", 6.8)
            c.drawCentredString(x0 + 13.5 * mm, y + self.rh - 5.0 * mm, tag)
            c.setFillColor(INK)
            c.setFont("Helvetica", 7.6)
            lines = body.split("\n")
            ty = y + self.rh - 9.2 * mm if len(lines) > 1 else y + 4.8 * mm
            for ln in lines:
                c.drawString(x0 + 3 * mm, ty, ln)
                ty -= 3.4 * mm
            c.setFillColor(GRAY)
            c.setFont("Helvetica-Oblique", 7.0)
            c.drawString(x0 + bw + 3 * mm, y + self.rh / 2 - 1 * mm, cite)
            if i < len(self.rows) - 1:
                ax = x0 + bw / 2
                c.setStrokeColor(ACCENT)
                c.setLineWidth(1.2)
                c.line(ax, y, ax, y - self.gap + 1.4 * mm)
                c.setFillColor(ACCENT)
                p = c.beginPath()
                p.moveTo(ax - 1.6 * mm, y - self.gap + 1.8 * mm)
                p.lineTo(ax + 1.6 * mm, y - self.gap + 1.8 * mm)
                p.lineTo(ax, y - self.gap)
                p.close()
                c.drawPath(p, stroke=0, fill=1)
            y -= self.rh + self.gap


# ---------- page furniture ----------
def on_page(canv, doc):
    canv.saveState()
    canv.setStrokeColor(LINE)
    canv.setLineWidth(0.6)
    canv.line(18 * mm, H - 14 * mm, W - 18 * mm, H - 14 * mm)
    canv.setFont("Helvetica-Bold", 7.5)
    canv.setFillColor(ACCENT)
    canv.drawString(18 * mm, H - 11.5 * mm, "SEMICON INDIA HACKATHON 2026  |  TRACK 1 (KLA)  |  PS-01: AI-BASED RESTORATION OF DEGRADED IMAGES")
    canv.setFont("Helvetica", 7.5)
    canv.setFillColor(GRAY)
    canv.drawRightString(W - 18 * mm, H - 11.5 * mm, "Team DataDrifters  |  Solution Architecture v2 (rectified)")
    canv.line(18 * mm, 13 * mm, W - 18 * mm, 13 * mm)
    canv.setFont("Helvetica", 7.5)
    canv.drawCentredString(W / 2, 9 * mm, f"Page {doc.page}")
    canv.restoreState()


doc = BaseDocTemplate(
    "public/KLA_PS1_Architecture.pdf",
    pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
    topMargin=20 * mm, bottomMargin=18 * mm,
    title="KLA PS-01 Unified Blind Restoration Network - Architecture",
)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=on_page)])

E = []

# ============ PAGE 1 ============
E.append(Spacer(1, 4 * mm))
E.append(P("Unified Blind Restoration Network (UBRN)", S_TITLE))
E.append(P("Single-model, single-pass restoration of jointly degraded semiconductor inspection images "
           "(speckle + Gaussian noise + downsampling, applied in randomized order).", S_SUB))
E.append(Spacer(1, 3 * mm))

E.append(P("1. Problem Formulation", S_H1))
E.append(P("Given a degraded grayscale image <b>Y = D(X)</b> where <b>D</b> is a randomized black-box composition of "
           "three degradations, learn the inverse mapping <b>f: Y &rarr; X&#770;</b> such that X&#770; matches the clean "
           "full-resolution ground truth. The problem is ill-posed (one Y maps to many plausible X), the composition "
           "order is randomized (per KLA: &ldquo;do not read into the order&rdquo;), and the test set contains "
           "out-of-distribution samples. Scoring is on three axes: restoration quality (SSIM, PSNR, LPIPS, L1/L2), "
           "end-to-end throughput on an NVIDIA H100 (including disk I/O), and training hygiene / reproducibility."))

E.append(P("2. Degradation Model (what the network must invert)", S_H2))
deg = Table(
    [
        [P("Degradation", S_CELLH), P("Nature", S_CELLH), P("Design consequence", S_CELLH)],
        [P("Speckle noise", S_CELLB), P("Multiplicative; scales with brightness; pushes pixels beyond [0,1]", S_CELL),
         P("Do NOT clip inputs (overshoot is signal). No log transform (see Flag F3).", S_CELL)],
        [P("Gaussian noise", S_CELLB), P("Additive, zero-mean, variable sigma (e.g. 0.05)", S_CELL),
         P("Blind handling via noise-level-map conditioning (FFDNet-style).", S_CELL)],
        [P("Downsampling", S_CELLB), P("512&rarr;256 or 256&rarr;128 (x2 per step)", S_CELL),
         P("Mandatory upsampling head; scale inferred from input shape at runtime.", S_CELL)],
    ],
    colWidths=[30 * mm, 62 * mm, 78 * mm],
)
deg.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), ACCENT_LT]),
    ("GRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
E.append(deg)

E.append(P("3. Design Principles", S_H1))
for t in [
    "<b>One network, one pass.</b> No routing/classifier branches: DnCNN proved a single model handles blind denoising + SISR + deblocking jointly; FFDNet proved a noise-level map replaces per-noise networks. One model load also maximizes H100 throughput.",
    "<b>End-to-end on intensity domain.</b> No homomorphic log transform: under mixed additive+multiplicative noise, pixels can be &le;0 and log() produces NaN &mdash; an instantly invalid output. ID-CNN and Dalsasso et al. both motivate direct end-to-end learning.",
    "<b>Train on the true joint degradation.</b> W2S shows denoise-then-SR cascades are inefficient and SR networks are highly noise-sensitive; the joint task must be trained jointly, with the degradation pipeline randomized exactly like KLA's black box.",
    "<b>Golden-ratio sizing.</b> ~5&ndash;15M parameters: largest model with measurable gains, distilled/pruned if throughput demands it.",
    "<b>Fidelity first, perception second.</b> Pixel/SSIM/frequency losses drive training; adversarial fine-tuning is an optional, ablated stage (W2S two-phase protocol) capped to avoid hallucinated structures that KLA explicitly warns about.",
]:
    E.append(bullet(t))

E.append(PageBreak())

# ============ PAGE 2 ============
E.append(P("4. End-to-End Pipeline", S_H1))
E.append(Pipeline())
E.append(P("Figure 1 &mdash; UBRN inference pipeline. Every stage is justified by one of the eight referenced papers.", S_CAP))

E.append(P("5. Module Specifications", S_H1))
mods = Table(
    [
        [P("Module", S_CELLH), P("Specification", S_CELLH), P("Source paper &amp; adaptation", S_CELLH)],
        [P("Noise estimator (optional)", S_CELLB),
         P("3x conv3x3 (16ch, ReLU) &rarr; per-pixel noise map M; concatenated with input as channel 2. Can be dropped for fully-blind training (ablation A1).", S_CELL),
         P("FFDNet: tunable noise map as input to a single network &mdash; adopted as learned conditioning instead of manual routing.", S_CELL)],
        [P("Shallow extraction", S_CELLB),
         P("Single 3x3 conv, 48 channels.", S_CELL),
         P("SwinIR 3-module template (shallow / deep / reconstruction).", S_CELL)],
        [P("Deep body", S_CELLB),
         P("4-level U-shaped encoder&ndash;decoder; [4,6,6,8] blocks of MDTA (multi-Dconv-head transposed attention, CxC channel attention, linear in HxW) + GDFN (gated depthwise-conv FFN); pixel-unshuffle down / pixel-shuffle up between levels; encoder&ndash;decoder skip concats.", S_CELL),
         P("Restormer: MDTA + GDFN blocks and U-shaped layout, scaled down to the 5&ndash;15M param zone.", S_CELL)],
        [P("Global skip", S_CELLB),
         P("Shallow features added to decoder output before the head; network learns the residual detail.", S_CELL),
         P("SwinIR Eq.5 long skip; DnCNN residual-learning principle.", S_CELL)],
        [P("Reconstruction head", S_CELLB),
         P("conv &rarr; PixelShuffle x2 (applied once or twice for x4) &rarr; conv3x3 &rarr; 1-channel output. Scale chosen at runtime from target/input shape ratio; both heads share the body.", S_CELL),
         P("SwinIR sub-pixel reconstruction; W2S/ESRGAN joint denoise+SR precedent.", S_CELL)],
        [P("Output guard", S_CELLB),
         P("torch.no_grad + eval(); nan_to_num &rarr; clip[0,1] &rarr; (H,W) float32 &rarr; same filename .npy.", S_CELL),
         P("KLA submission contract (valid range, no NaN/Inf, exact naming).", S_CELL)],
    ],
    colWidths=[28 * mm, 82 * mm, 60 * mm],
)
mods.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), ACCENT_LT]),
    ("GRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
E.append(mods)

E.append(PageBreak())

# ============ PAGE 3 ============
E.append(P("6. Loss Function", S_H1))
E.append(P("Primary (fidelity phase &mdash; used for the submitted model):"))
E.append(P("L = L_Char + 0.2 * L_SSIM + 0.1 * L_LPIPS + 0.05 * L_FFT", S_CODE))
for t in [
    "<b>Charbonnier L1</b> (&radic;(&Delta;&sup2;+&epsilon;&sup2;), &epsilon;=1e-3) &mdash; stable pixel fidelity (SwinIR Eq.7).",
    "<b>SSIM loss</b> &mdash; directly optimizes an officially scored metric.",
    "<b>LPIPS loss</b> &mdash; perceptual metric is officially scored; VGG weights bundled locally (offline rule); training-only, never loaded in run.py.",
    "<b>FFT-L1 (frequency) loss</b> &mdash; semiconductor patterns are periodic; KLA explicitly flagged frequency-domain signal as a promising axis.",
    "<b>Optional phase 2 (ablation A3):</b> relativistic-discriminator fine-tune, lambda_adv &le; 0.005, following the W2S two-phase protocol. Submitted only if it improves LPIPS without degrading PSNR/SSIM; discriminator never ships in run.py.",
]:
    E.append(bullet(t))

E.append(P("7. Training Strategy", S_H1))
for t in [
    "<b>Randomized degradation synthesis (core OOD weapon).</b> Re-implement KLA's black box as the on-the-fly augmenter: random order of {Gaussian sigma in [0.01,0.1], speckle strength, blur, downsample x2/x4}, plus the 8-config flip/rotation augmentation (W2S Sec. 2.1). Never train on a fixed degradation order.",
    "<b>Mandatory sanity check first:</b> overfit 1&ndash;2 image pairs to perfect reconstruction; failure indicates a loader / normalization / loss bug (KLA playbook requirement).",
    "<b>Progressive learning:</b> start with small patches (128) &amp; large batches, finish with large patches (256&ndash;384) &amp; small batches (Restormer protocol) &mdash; improves large-context generalization.",
    "<b>Inputs fed raw and unclipped;</b> clipping applied only to final outputs.",
    "<b>Validation:</b> held-out split scored with PSNR + SSIM + LPIPS every epoch; visual inspection of outputs each milestone (high PSNR can coexist with hallucinated structures).",
    "<b>Reproducibility:</b> fixed seeds, config-file-driven runs, pip-frozen requirements.txt, optional Dockerfile (training-hygiene axis).",
]:
    E.append(bullet(t))

E.append(P("8. Inference &amp; Throughput Engineering (H100, end-to-end incl. disk I/O)", S_H1))
for t in [
    "run.py &lt;input_dir&gt; &lt;output_dir&gt;: standalone .py (not a notebook); creates output dir; iterates sorted *.npy; saves identical filenames; weights loaded from local models/ folder; zero network access.",
    "FP16 (model.half) + torch.compile + channels_last memory format.",
    "Batched inference: group equal-shape .npy files into batches per forward pass.",
    "Background-thread prefetching of .npy files &mdash; disk I/O is inside the scored window.",
    "Optional: ONNX / TensorRT export if time permits (ablation A4).",
    "Optional self-ensemble (x8 flips/rotations) behind a flag &mdash; enabled only if the measured speed budget allows.",
]:
    E.append(bullet(t))

E.append(PageBreak())

# ============ PAGE 4 ============
E.append(P("9. Rectified Design Flags (audit trail)", S_H1))
flags = Table(
    [
        [P("Flag", S_CELLH), P("Original design", S_CELLH), P("Risk", S_CELLH), P("Rectification", S_CELLH)],
        [P("F1", S_CELLB), P("No super-resolution stage (DnCNN/FFDNet are same-resolution)", S_CELL),
         P("Every output at wrong resolution &rarr; automatic fail", S_CELL),
         P("PixelShuffle reconstruction head (SwinIR); joint training on noise+SR (W2S)", S_CELL)],
        [P("F2", S_CELLB), P("Hard routing: noise classifier picks DnCNN / FFDNet / log branch", S_CELL),
         P("Misrouting on mixed &amp; OOD inputs; 3 model loads hurt throughput", S_CELL),
         P("Single blind network + FFDNet-style noise-map conditioning", S_CELL)],
        [P("F3", S_CELLB), P("Homomorphic log transform for speckle", S_CELL),
         P("Mixed additive noise means pixels can be &le;0 so log = NaN (invalid output); log-speckle is biased non-Gaussian (Fisher&ndash;Tippett)", S_CELL),
         P("In-network despeckling on intensity domain (ID-CNN, Dalsasso et al.)", S_CELL)],
        [P("F4", S_CELLB), P("Conditional GAN cascade (\"if pixel loss low then adversarial\")", S_CELL),
         P("Training instability; hallucinated structures; PSNR/SSIM penalty", S_CELL),
         P("Fidelity-first training; optional capped adversarial fine-tune as ablation (W2S two-phase)", S_CELL)],
        [P("F5", S_CELLB), P("Inputs normalized/clipped to [0,1]", S_CELL),
         P("Destroys speckle-overshoot signal KLA told teams to keep", S_CELL),
         P("Raw inputs; clip only final outputs", S_CELL)],
        [P("F6", S_CELLB), P("LPIPS/VGG auto-download at first use", S_CELL),
         P("Evaluation machine is offline &rarr; crash", S_CELL),
         P("Perceptual nets are training-only; any inference weights bundled in models/", S_CELL)],
    ],
    colWidths=[10 * mm, 47 * mm, 52 * mm, 61 * mm],
)
flags.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), WARN),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#faf3ee")]),
    ("GRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
E.append(flags)

E.append(P("10. Ablation Plan (training-hygiene evidence)", S_H1))
abl = Table(
    [
        [P("ID", S_CELLH), P("Ablation", S_CELLH), P("Question answered", S_CELLH)],
        [P("A1", S_CELLB), P("With vs. without noise-estimator conditioning", S_CELL), P("Does explicit conditioning beat fully-blind?", S_CELL)],
        [P("A2", S_CELLB), P("MDTA/GDFN body (Restormer) vs. RSTB windowed body (SwinIR)", S_CELL), P("Channel vs. windowed-spatial attention: PSNR-vs-throughput trade-off", S_CELL)],
        [P("A3", S_CELLB), P("Fidelity-only vs. + adversarial fine-tune", S_CELL), P("Does GAN help LPIPS without hurting PSNR/SSIM?", S_CELL)],
        [P("A4", S_CELLB), P("FP32 vs. FP16 vs. TensorRT", S_CELL), P("Throughput gain vs. quality drift", S_CELL)],
        [P("A5", S_CELLB), P("Fixed vs. randomized degradation order in training", S_CELL), P("Quantifies the OOD-generalization win", S_CELL)],
    ],
    colWidths=[10 * mm, 88 * mm, 72 * mm],
)
abl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), OK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#eef6f0")]),
    ("GRID", (0, 0), (-1, -1), 0.5, LINE),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
E.append(abl)

E.append(P("11. References (the 8 uploaded papers)", S_H1))
refs = [
    "Zhang et al., \"Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising\" (DnCNN), IEEE TIP 2017 &mdash; residual learning; single model for blind denoising + SISR + deblocking.",
    "Zhang et al., \"FFDNet: Toward a Fast and Flexible Solution for CNN-based Image Denoising\", IEEE TIP 2018 &mdash; noise-level map conditioning of a single network.",
    "Wang et al., \"SAR Image Despeckling Using a Convolutional Neural Network\" (ID-CNN), IEEE SPL 2017 &mdash; end-to-end despeckling on intensity domain, rejecting log-domain pipelines.",
    "Dalsasso et al., 2020 (arXiv:2006.15559) &mdash; Fisher&ndash;Tippett analysis: log-speckle is non-Gaussian and biased.",
    "Zhu et al., \"Deep Learning Meets SAR\", 2020 (arXiv:2006.10027) &mdash; learned despeckling surpasses handcrafted homomorphic filtering.",
    "Zamir et al., \"Restormer: Efficient Transformer for High-Resolution Image Restoration\", CVPR 2022 (arXiv:2111.09881) &mdash; MDTA + GDFN blocks, progressive learning.",
    "Liang et al., \"SwinIR: Image Restoration Using Swin Transformer\", ICCVW 2021 (arXiv:2108.10257) &mdash; shallow/deep/reconstruction template, PixelShuffle head, Charbonnier loss.",
    "Zhou et al., \"W2S: Joint Denoising and Super-Resolution\" (core1 paper) &mdash; joint noise+SR must be trained jointly; two-phase GAN protocol and its PSNR trade-off.",
]
for i, r in enumerate(refs, 1):
    E.append(bullet(f"[{i}] {r}"))

doc.build(E)
print("PDF written: public/KLA_PS1_Architecture.pdf")
