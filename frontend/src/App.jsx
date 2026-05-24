import { useState, useEffect } from 'react'
import QueryInput from './components/QueryInput'
import AnswerDisplay from './components/AnswerDisplay'
import SourcesPanel from './components/SourcesPanel'
import DebugPanel from './components/DebugPanel'

const API_BASE = '/api'

function App() {
  const [response, setResponse] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  const [uploadStatus, setUploadStatus] = useState(null)

  // Health check on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: 'offline', document_count: 0 }))
  }, [])

  const handleQuery = async (query) => {
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResponse(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (file) => {
    setUploadStatus(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setUploadStatus({ type: 'success', message: data.message })

      // Refresh health to update doc count
      fetch(`${API_BASE}/health`)
        .then(r => r.json())
        .then(d => setHealth(d))
        .catch(() => {})
    } catch (err) {
      setUploadStatus({ type: 'error', message: err.message })
    }
  }

  const healthStatus = health?.status === 'healthy'
    ? 'healthy'
    : health?.status === 'offline'
    ? 'offline'
    : 'degraded'

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-icon">⚡</div>
        <h1>
          RAG System
          <span>Debug Console</span>
        </h1>
        <div className="health-badge">
          <div className={`health-dot ${healthStatus}`} />
          {healthStatus === 'healthy'
            ? `${health?.document_count || 0} docs indexed`
            : healthStatus}
        </div>
      </header>

      {/* Query + Upload */}
      <QueryInput
        onQuery={handleQuery}
        onUpload={handleUpload}
        loading={loading}
        uploadStatus={uploadStatus}
      />

      {/* Error */}
      {error && (
        <div className="error-banner">
          ⚠ {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="spinner" />
          Running pipeline: expand → retrieve → rerank → generate → analyze...
        </div>
      )}

      {/* Results */}
      {response && !loading && (
        <>
          <AnswerDisplay
            answer={response.answer}
            latency={response.total_latency_ms}
            confidence={response.confidence_score}
            confidenceLabel={response.confidence_label}
            failureClass={response.debug?.failure_class}
          />
          <SourcesPanel sources={response.sources} />
          <DebugPanel debug={response.debug} />
        </>
      )}

      {/* Empty state */}
      {!response && !loading && !error && (
        <div className="empty-state">
          <div className="icon">🔍</div>
          <p>
            Upload a document (PDF, Markdown, or text) and ask a question.
            The debug panel will show retrieval scores, reranking, confidence, and the full prompt.
          </p>
        </div>
      )}
    </div>
  )
}

export default App
