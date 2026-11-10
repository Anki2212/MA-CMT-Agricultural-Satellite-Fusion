export default function About() {
  return (
    <div className="max-w-3xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">About</h1>
        <p className="text-slate-500 mt-1 text-sm">
          Research context, dataset, architecture and methodology.
        </p>
      </div>

      <Section title="Research Problem">
        <p>
          Agricultural remote sensing relies on multi-sensor satellite time-series data combining 
          Synthetic Aperture Radar (Sentinel-1) and multispectral optical (Sentinel-2) observations. 
          In practice, data are frequently unavailable due to cloud cover, revisit timing mismatches, 
          and acquisition failures — creating irregular missing patterns that degrade model performance.
        </p>
        <p className="mt-2">
          Existing methods either drop missing observations, impute them, or ignore modality-specific 
          availability — all of which lose information or introduce artifacts. This project proposes 
          <strong> MA-CMT</strong>, a Transformer-based model that explicitly conditions fusion on 
          binary availability masks.
        </p>
      </Section>

      <Section title="Objectives">
        <ul className="list-disc list-inside space-y-1 text-sm text-slate-700">
          <li>Build a dataset pipeline for CropHarvest combining S1 + S2 at 12-month resolution.</li>
          <li>Design MA-CMT with availability-guided gated cross-modal fusion.</li>
          <li>Evaluate performance under random, sensor-level and consecutive missing data.</li>
          <li>Compare MA-CMT against three baselines: S1-only, S2-only, naïve concatenation.</li>
          <li>Demonstrate robustness at 0%, 10%, 20%, 30% and 50% missing rates.</li>
        </ul>
      </Section>

      <Section title="Dataset">
        <div className="grid grid-cols-2 gap-4">
          {[
            ['Name', 'CropHarvest'],
            ['Labels file', 'labels.geojson (113,893 polygons)'],
            ['Feature archive', 'features.tar.gz (87,464 HDF5 files)'],
            ['Matched labeled', '42,603 samples with classification labels'],
            ['Classes', '10 FAO-based crop types'],
            ['Time steps', '12 months'],
            ['Features used', '13 (S1: VV, VH · S2: B2–B12)'],
            ['Excluded', 'Weather, elevation, slope, NDVI'],
          ].map(([k, v]) => (
            <div key={k} className="bg-slate-50 rounded-lg p-3 border border-slate-100">
              <p className="text-xs font-semibold text-slate-500 uppercase">{k}</p>
              <p className="text-sm text-slate-800 mt-0.5">{v}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Architecture — MA-CMT">
        <div className="space-y-3 text-sm text-slate-700">
          <ArchStep n={1} title="Per-modal Encoders">
            Separate linear projection + LayerNorm + GELU for S1 (2→128 dim) and S2 (11→128 dim). 
            Encoder outputs are zeroed for missing months.
          </ArchStep>
          <ArchStep n={2} title="Availability-Guided Gated Fusion">
            A gate network accepts the concatenation of [h_S1, h_S2, mask_S1, mask_S2] 
            and produces two scalar gates (via Sigmoid). The fused representation is 
            g₁·h_S1 + g₂·h_S2, projected back to d_model. Missing sensors automatically 
            contribute zero because their encoder output is zeroed before gating.
          </ArchStep>
          <ArchStep n={3} title="Temporal Transformer Encoder">
            3-layer Transformer encoder with sinusoidal positional encoding operates 
            over 12 monthly tokens. Fully-missing months are excluded via key-padding masks.
          </ArchStep>
          <ArchStep n={4} title="Classification Head">
            Mean pooling over valid time steps → LayerNorm → Linear(128, 64) → GELU → 
            Dropout → Linear(64, 10 classes).
          </ArchStep>
        </div>
      </Section>

      <Section title="Methodology">
        <ul className="list-disc list-inside space-y-1 text-sm text-slate-700">
          <li>Stratified 70/15/15 train/val/test split.</li>
          <li>Per-feature Z-score normalization computed on training set only.</li>
          <li>Missing data simulated post-normalization; original data unchanged.</li>
          <li>Weighted cross-entropy loss to handle class imbalance.</li>
          <li>AdamW optimizer with cosine annealing LR schedule.</li>
          <li>Early stopping on validation accuracy (patience = 10).</li>
          <li>Evaluation: accuracy, macro precision, recall, F1, confusion matrix.</li>
        </ul>
      </Section>

      <Section title="Technologies">
        <div className="flex flex-wrap gap-2">
          {[
            'Python 3.12', 'PyTorch 2.x', 'FastAPI', 'CropHarvest',
            'React 18', 'TypeScript', 'Vite', 'Tailwind CSS', 'Recharts',
            'NumPy', 'scikit-learn', 'h5py', 'Pandas', 'pytest',
          ].map(t => (
            <span key={t} className="px-2 py-1 rounded bg-green-50 text-green-800 text-xs font-medium border border-green-100">
              {t}
            </span>
          ))}
        </div>
      </Section>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide mb-3">{title}</h2>
      {children}
    </div>
  )
}

function ArchStep({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-green-700 text-white text-xs font-bold flex items-center justify-center mt-0.5">
        {n}
      </div>
      <div>
        <p className="font-semibold text-slate-800">{title}</p>
        <p className="text-slate-600 text-sm mt-0.5">{children}</p>
      </div>
    </div>
  )
}
