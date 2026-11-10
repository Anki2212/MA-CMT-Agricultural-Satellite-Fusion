import { useEffect, useState } from 'react'
import api from '../api'
import type { DatasetInfo, HealthStatus, ModelStatus } from '../api'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const CLASS_COLORS = [
  '#166534', '#15803d', '#16a34a', '#22c55e', '#4ade80',
  '#86efac', '#a7f3d0', '#6ee7b7', '#34d399', '#10b981',
]

export default function Dashboard() {
  const [info, setInfo] = useState<DatasetInfo | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [models, setModels] = useState<Record<string, ModelStatus> | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      api.get('/dataset/info'),
      api.get('/health'),
      api.get('/models/status'),
    ])
      .then(([infoRes, healthRes, modelsRes]) => {
        setInfo(infoRes.data)
        setHealth(healthRes.data)
        setModels(modelsRes.data)
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <LoadingState />
  if (error) return <ErrorState message={error} />

  const classChartData = info
    ? Object.entries(info.class_distribution)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
    : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-500 mt-1 text-sm">
          CropHarvest dataset overview and model availability
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Samples" value={info?.total_samples.toLocaleString() ?? '—'} />
        <StatCard label="Training" value={info?.train_samples.toLocaleString() ?? '—'} sub="70%" />
        <StatCard label="Validation" value={info?.val_samples.toLocaleString() ?? '—'} sub="15%" />
        <StatCard label="Test" value={info?.test_samples.toLocaleString() ?? '—'} sub="15%" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Crop Classes" value={String(info?.n_classes ?? '—')} />
        <StatCard label="Time Steps" value={`${info?.n_months ?? '—'} months`} />
        <StatCard label="Total Features" value={String(info?.n_features ?? '—')} />
        <StatCard label="Device" value={health?.device ?? '—'} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Class distribution chart */}
        <Card title="Class Distribution">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={classChartData} layout="vertical" margin={{ left: 16, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => (v as number).toLocaleString()} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {classChartData.map((_, i) => (
                  <Cell key={i} fill={CLASS_COLORS[i % CLASS_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Model status */}
        <Card title="Model Checkpoints">
          <div className="space-y-3 mt-2">
            {models && Object.entries(models).map(([name, status]) => (
              <div key={name} className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-100">
                <div>
                  <span className="font-medium text-sm text-slate-800">{MODEL_LABELS[name] ?? name}</span>
                  {status.available && status.epoch !== undefined && (
                    <span className="ml-2 text-xs text-slate-400">epoch {status.epoch}</span>
                  )}
                </div>
                {status.available ? (
                  <div className="text-right">
                    <span className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                      Ready
                    </span>
                    {status.val_acc !== undefined && (
                      <div className="text-xs text-slate-500 mt-0.5">
                        val acc: {(status.val_acc * 100).toFixed(1)}%
                      </div>
                    )}
                  </div>
                ) : (
                  <span className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700">
                    Not trained
                  </span>
                )}
              </div>
            ))}
          </div>
          {!models || Object.values(models).every(m => !m.available) ? (
            <p className="text-sm text-amber-700 mt-3 p-3 bg-amber-50 rounded-lg border border-amber-200">
              No checkpoints found. Run <code className="font-mono text-xs bg-amber-100 px-1 rounded">python ml/train.py --model macmt</code> to train.
            </p>
          ) : null}
        </Card>
      </div>

      {/* Features */}
      <Card title="Features">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              Sentinel-1 (SAR)
            </h3>
            <div className="flex flex-wrap gap-2">
              {info?.s1_features.map(f => (
                <span key={f} className="px-2 py-1 rounded bg-blue-50 text-blue-700 text-xs font-medium border border-blue-100">{f}</span>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
              Sentinel-2 (Optical)
            </h3>
            <div className="flex flex-wrap gap-2">
              {info?.s2_features.map(f => (
                <span key={f} className="px-2 py-1 rounded bg-emerald-50 text-emerald-700 text-xs font-medium border border-emerald-100">{f}</span>
              ))}
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}

const MODEL_LABELS: Record<string, string> = {
  macmt:  'MA-CMT (Proposed)',
  s1only: 'S1-Only Baseline',
  s2only: 'S2-Only Baseline',
  concat: 'Concat Fusion Baseline',
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
      <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-slate-900 mt-1">{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
    </div>
  )
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
      <h2 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">{title}</h2>
      {children}
    </div>
  )
}

function LoadingState() {
  return (
    <div className="flex items-center justify-center h-64 text-slate-400">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-green-600 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
        <p className="text-sm">Loading…</p>
      </div>
    </div>
  )
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm">
      <strong>Error:</strong> {message}
      <p className="mt-1 text-red-500">Ensure the FastAPI backend is running on port 8000.</p>
    </div>
  )
}
