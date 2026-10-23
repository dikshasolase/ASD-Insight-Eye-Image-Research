import { useEffect, useRef, useState } from 'react'
import {
  Activity, ArrowRight, BrainCircuit, CircleAlert, Eye, FileImage,
  FlaskConical, History, ImagePlus, LoaderCircle, ShieldCheck, Sparkles,
  UploadCloud, X,
} from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const MAX_BYTES = 8 * 1024 * 1024
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

export default function App() {
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [history, setHistory] = useState([])
  const [dragging, setDragging] = useState(false)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  function chooseFile(nextFile) {
    setError('')
    setResult(null)
    if (!nextFile) return
    if (!ACCEPTED.includes(nextFile.type)) return setError('Choose a JPG, PNG, or WebP image.')
    if (nextFile.size > MAX_BYTES) return setError('The image must be smaller than 8 MB.')
    if (preview) URL.revokeObjectURL(preview)
    setFile(nextFile)
    setPreview(URL.createObjectURL(nextFile))
  }

  async function analyze() {
    if (!file) return setError('Select an image first.')
    setBusy(true); setError(''); setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Analysis failed.')
      setResult(data)
      setHistory(items => [{ name: file.name, output: data.research_output, time: new Date().toLocaleTimeString() }, ...items].slice(0, 5))
    } catch (err) {
      setError(err.message || 'Could not reach the API. Check that FastAPI is running.')
    } finally { setBusy(false) }
  }

  function clearImage() {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null); setPreview(''); setResult(null); setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const scores = result?.class_scores_percent || {}
  return (
    <div className="min-h-screen bg-midnight text-slate-100">
      <header className="border-b border-white/[0.08] bg-midnight/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-8">
          <a href="#" className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-2xl border border-indigo-300/20 bg-indigo-400/10 text-indigo-200"><BrainCircuit size={22} /></div>
            <div><p className="font-semibold tracking-tight">ASD <span className="text-indigo-300">Insight</span></p><p className="text-[10px] uppercase tracking-[.2em] text-slate-500">Image intelligence lab</p></div>
          </a>
          <div className="hidden items-center gap-2 rounded-full border border-amber-200/15 bg-amber-200/[0.06] px-3 py-2 text-xs text-amber-100 sm:flex"><FlaskConical size={14} /> Experimental research</div>
          <span className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-slate-400">Prototype v1.0</span>
        </div>
      </header>

      <main className="drop-grid mx-auto max-w-7xl px-5 py-10 lg:px-8 lg:py-14">
        <section className="grid gap-10 lg:grid-cols-[1.05fr_.95fr] lg:items-center">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-indigo-300/20 bg-indigo-300/[0.07] px-3 py-1.5 text-xs text-indigo-200"><Sparkles size={14} /> Research-grade workflow</div>
            <h1 className="max-w-2xl text-4xl font-semibold leading-[1.08] tracking-tight sm:text-5xl lg:text-6xl">Explore image patterns.<span className="mt-2 block bg-gradient-to-r from-indigo-200 via-violet-200 to-teal-200 bg-clip-text text-transparent">Understand the model.</span></h1>
            <p className="mt-6 max-w-xl text-base leading-7 text-slate-400">Upload a labeled eye-region research image and inspect the output of an experimental image classifier, with clear limitations and transparent class scores.</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <div className="flex items-center gap-2 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-slate-300"><ShieldCheck size={17} className="text-teal-300" /> Input checks</div>
              <div className="flex items-center gap-2 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-slate-300"><Activity size={17} className="text-indigo-300" /> Model scores</div>
              <div className="flex items-center gap-2 rounded-xl border border-white/[0.08] bg-white/[0.03] px-4 py-3 text-sm text-slate-300"><Eye size={17} className="text-violet-300" /> Eye-image study</div>
            </div>
          </div>

          <div className="rounded-3xl border border-white/10 bg-panel/90 p-5 shadow-glow sm:p-7">
            <div className="flex items-start justify-between gap-4"><div><p className="text-xs font-medium uppercase tracking-[.18em] text-indigo-200">Image workspace</p><h2 className="mt-2 text-xl font-semibold">Upload eye-region image</h2><p className="mt-1 text-sm text-slate-400">JPG, PNG or WebP · Up to 8 MB</p></div><div className="rounded-xl border border-white/10 bg-white/[0.04] p-3 text-indigo-200"><FileImage size={22} /></div></div>
            <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={e => chooseFile(e.target.files?.[0])} />
            <div role="button" tabIndex={0} onClick={() => !preview && inputRef.current?.click()} onKeyDown={e => { if ((e.key === 'Enter' || e.key === ' ') && !preview) inputRef.current?.click() }} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files?.[0]) }} className={`mt-6 flex min-h-64 flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed p-4 text-center transition ${dragging ? 'border-indigo-300 bg-indigo-300/10' : 'border-slate-600/70 bg-slate-950/40'} ${preview ? 'cursor-default' : 'cursor-pointer hover:border-indigo-300/70 hover:bg-indigo-300/[0.04]'}`}>
              {preview ? <div className="relative w-full"><img src={preview} alt="Selected research image preview" className="mx-auto max-h-72 rounded-xl object-contain" /><button onClick={e => { e.stopPropagation(); clearImage() }} className="absolute right-1 top-1 rounded-lg border border-white/10 bg-midnight/90 p-2 text-slate-300 hover:text-white" aria-label="Remove image"><X size={16} /></button><p className="mt-3 truncate text-sm text-slate-300">{file?.name}</p></div> : <><div className="mb-4 grid h-14 w-14 place-items-center rounded-2xl border border-indigo-300/20 bg-indigo-300/[0.08] text-indigo-200"><UploadCloud size={25} /></div><p className="font-medium">Drop your image here</p><p className="mt-1 text-sm text-slate-500">or click to browse your files</p><span className="mt-4 inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.05] px-4 py-2.5 text-sm text-slate-200"><ImagePlus size={16} /> Choose image</span></>}
            </div>
            {error && <div role="alert" className="mt-4 flex gap-2 rounded-xl border border-rose-300/20 bg-rose-400/[0.07] p-3 text-sm text-rose-200"><CircleAlert size={18} className="shrink-0" />{error}</div>}
            <button onClick={analyze} disabled={!file || busy} className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-indigo-300 px-5 py-3.5 font-semibold text-slate-950 transition hover:bg-indigo-200 disabled:cursor-not-allowed disabled:opacity-40">{busy ? <><LoaderCircle size={18} className="animate-spin" /> Analyzing image…</> : <>Run experimental analysis <ArrowRight size={18} /></>}</button>
            <p className="mt-3 text-center text-xs leading-5 text-slate-500">Upload validation checks file integrity only. This prototype does not confirm that an image contains an eye.</p>
          </div>
        </section>

        <section className="mt-10 grid gap-5 lg:grid-cols-[1fr_320px]">
          <div className="rounded-2xl border border-white/[0.08] bg-panel/70 p-5 sm:p-6">
            <div className="flex items-center gap-2"><Activity size={18} className="text-indigo-300" /><h2 className="font-semibold">Analysis output</h2></div>
            {!result ? <div className="mt-5 flex min-h-36 flex-col items-center justify-center rounded-xl border border-white/[0.06] bg-black/10 p-5 text-center"><div className="mb-3 rounded-xl bg-white/[0.04] p-3 text-slate-500"><BrainCircuit size={22} /></div><p className="text-sm text-slate-400">Your model response will appear here</p><p className="mt-1 text-xs text-slate-600">Train the model and start the API before analyzing.</p></div> : <div className="mt-5 space-y-4"><div className="rounded-xl border border-indigo-300/15 bg-indigo-300/[0.05] p-4"><p className="text-xs uppercase tracking-wider text-slate-400">Experimental model output</p><p className="mt-2 text-2xl font-semibold text-indigo-200">{result.research_output}</p><p className="mt-2 text-xs leading-5 text-slate-400">{result.notice}</p></div><div><p className="mb-3 text-sm font-medium text-slate-300">Model class scores</p>{Object.entries(scores).map(([label, score]) => <div key={label} className="mb-3"><div className="mb-1.5 flex justify-between gap-3 text-sm"><span className="text-slate-300">{label}</span><span className="tabular-nums text-slate-400">{Number(score).toFixed(2)}%</span></div><div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-gradient-to-r from-indigo-400 to-teal-300" style={{ width: `${Math.min(100, Math.max(0, Number(score)))}%` }} /></div></div>)}</div><p className="text-xs text-slate-500">These scores are not calibrated clinical probabilities and must not be used for diagnosis or decisions about care.</p></div>}
          </div>
          <aside className="rounded-2xl border border-white/[0.08] bg-panel/70 p-5"><div className="flex items-center gap-2"><History size={18} className="text-teal-300" /><h2 className="font-semibold">Session history</h2></div>{history.length === 0 ? <p className="mt-4 text-sm leading-6 text-slate-500">Completed analyses in this browser session appear here. History is not saved to a database.</p> : <ul className="mt-4 space-y-3">{history.map((item, i) => <li key={`${item.name}-${i}`} className="rounded-xl border border-white/[0.07] bg-black/10 p-3"><p className="truncate text-sm text-slate-200">{item.name}</p><p className="mt-1 text-xs text-indigo-200">{item.output}</p><p className="mt-1 text-xs text-slate-500">{item.time}</p></li>)}</ul>}</aside>
        </section>

        <section className="mt-8 grid gap-4 md:grid-cols-3">
          {[{ icon: ShieldCheck, title: 'Input integrity', desc: 'Checks allowed file types, image validity and upload size.' }, { icon: BrainCircuit, title: 'Experimental classifier', desc: 'Loads a trained ResNet18 baseline and reports class scores.' }, { icon: FlaskConical, title: 'Research boundaries', desc: 'The output is not a diagnosis and cannot rule out ASD.' }].map(({ icon: Icon, title, desc }) => <div key={title} className="rounded-2xl border border-white/[0.07] bg-white/[0.025] p-5"><Icon size={20} className="text-indigo-200" /><h3 className="mt-3 font-medium">{title}</h3><p className="mt-2 text-sm leading-6 text-slate-500">{desc}</p></div>)}
        </section>
        <footer className="mt-10 flex flex-col gap-3 border-t border-white/[0.08] pt-6 text-xs leading-5 text-slate-500 sm:flex-row sm:items-start sm:justify-between"><p>ASD Insight · AI-Driven Image Intelligence and Pattern Recognition System</p><p className="max-w-2xl">Research prototype only. Eye-region image classification is not a validated standalone ASD diagnostic method. Do not use outputs for medical decisions.</p></footer>
      </main>
    </div>
  )
}
