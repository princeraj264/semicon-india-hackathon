const pipeline = [
  { step: "Input", detail: "degraded .npy (raw, unclipped)" },
  { step: "Noise Estimator", detail: "3-layer CNN noise map (FFDNet)" },
  { step: "Shallow Conv", detail: "3\u00d73 feature extraction (SwinIR)" },
  { step: "Restormer-lite Body", detail: "4-level U-shape, MDTA + GDFN, ~6.5M params" },
  { step: "Global Skip", detail: "low frequencies bypass deep body" },
  { step: "PixelShuffle Head", detail: "\u00d72 super-resolution (128\u2192256, per official dataset)" },
  { step: "Output Guard", detail: "nan_to_num \u2192 clip[0,1] \u2192 (H,W) float32" },
]

const files = [
  { path: "run.py", desc: "Rule-compliant inference: run.py <input_dir> <output_dir>. Executed as-is by KLA on the H100." },
  { path: "train.py", desc: "Training with randomized degradation synthesis, --sanity overfit mode, checkpointing to models/." },
  { path: "src/model.py", desc: "UnifiedRestorationNet \u2014 noise-conditioned Restormer-lite with PixelShuffle SR head." },
  { path: "src/degradations.py", desc: "Randomized black-box synthesizer: Gaussian + speckle + blur + downsample in random order." },
  { path: "src/losses.py", desc: "Charbonnier + SSIM + LPIPS + FFT composite loss, PSNR/SSIM metrics." },
  { path: "src/dataset.py", desc: "Clean-image dataset with on-the-fly degradation pairing and 8-config augmentation." },
  { path: "requirements.txt", desc: "Pinned dependencies for offline reproduction." },
  { path: "docs/KLA_PS1_Architecture.pdf", desc: "Full architecture document with flag audit and 8-paper citations." },
]

const dataset = [
  { label: "train.zip (876 MB)", detail: "3,200 pairs \u2014 GT 256\u00d7256 in [0,1] + NoisyLR 128\u00d7128 unclipped (observed [-0.09, 1.88]), paired by filename" },
  { label: "Test_NoisyLR.zip (22 MB)", detail: "400 test images \u2014 128\u00d7128 float32, unclipped (observed [-0.22, 2.16])" },
  { label: "Scale factor", detail: "fixed \u00d72 (128 \u2192 256), verified from downloaded data" },
]

const checks = [
  "Model forward: \u00d72 (64\u2192128) and \u00d74 (32\u2192128) verified",
  "Sanity training on OFFICIAL pairs (train.py --kla --sanity): loss decreasing, PSNR climbing",
  "run.py on the REAL 400-image test set: 400/400 outputs compliant (256\u00d7256 float32, [0,1], finite, matching filenames)",
  "Composite loss backward pass verified",
]

export default function Page() {
  return (
    <main className="min-h-screen bg-background text-foreground font-mono">
      <div className="mx-auto max-w-3xl px-6 py-12 flex flex-col gap-10">
        <header className="flex flex-col gap-2 border-b border-border pb-6">
          <p className="text-xs uppercase tracking-widest text-muted-foreground">
            Team DataDrifters &mdash; SEMICON India Hackathon 2026 &mdash; Track 1 (KLA) &mdash; PS 1
          </p>
          <h1 className="text-2xl font-bold text-balance">
            Unified Blind Restoration Network
          </h1>
          <p className="text-sm text-muted-foreground leading-relaxed">
            AI-based restoration of degraded semiconductor inspection images:
            one model, one pass &mdash; speckle + Gaussian noise removal and
            super-resolution, engineered for H100 throughput.
          </p>
        </header>

        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-bold uppercase tracking-wider">Pipeline</h2>
          <ol className="flex flex-col">
            {pipeline.map((p, i) => (
              <li key={p.step} className="flex gap-4 items-baseline py-2 border-b border-border/50 last:border-0">
                <span className="text-xs text-muted-foreground w-5 shrink-0">{String(i + 1).padStart(2, "0")}</span>
                <span className="text-sm font-semibold w-40 shrink-0">{p.step}</span>
                <span className="text-xs text-muted-foreground">{p.detail}</span>
              </li>
            ))}
          </ol>
        </section>

        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-bold uppercase tracking-wider">Repository &mdash; kla-ps1/</h2>
          <ul className="flex flex-col gap-2">
            {files.map((f) => (
              <li key={f.path} className="flex flex-col gap-0.5">
                <code className="text-sm font-semibold text-primary">{f.path}</code>
                <span className="text-xs text-muted-foreground leading-relaxed">{f.desc}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-bold uppercase tracking-wider">Official Dataset</h2>
          <ul className="flex flex-col gap-2">
            {dataset.map((d) => (
              <li key={d.label} className="flex flex-col gap-0.5">
                <span className="text-sm font-semibold">{d.label}</span>
                <span className="text-xs text-muted-foreground leading-relaxed">{d.detail}</span>
              </li>
            ))}
          </ul>
          <a
            href="https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary underline underline-offset-4"
          >
            Google Drive dataset folder
          </a>
        </section>

        <section className="flex flex-col gap-3">
          <h2 className="text-sm font-bold uppercase tracking-wider">Verified Checks</h2>
          <ul className="flex flex-col gap-1.5">
            {checks.map((c) => (
              <li key={c} className="flex gap-2 items-baseline text-xs text-muted-foreground">
                <span aria-hidden="true" className="text-primary">[x]</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </section>

        <footer className="border-t border-border pt-6 flex flex-col gap-2">
          <a
            href="/KLA_PS1_Architecture.pdf"
            className="text-sm font-semibold text-primary underline underline-offset-4"
          >
            View full architecture PDF
          </a>
          <p className="text-xs text-muted-foreground">
            Deadline: Round 1 submission &mdash; 16 Aug 2026 &middot; File name: DataDrifters_KLA_PS01.pdf
          </p>
        </footer>
      </div>
    </main>
  )
}
