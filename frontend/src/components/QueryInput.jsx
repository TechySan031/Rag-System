import { useState, useRef } from 'react'

function QueryInput({ onQuery, onUpload, loading, uploadStatus }) {
  const [query, setQuery] = useState('')
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim() && !loading) {
      onQuery(query.trim())
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) onUpload(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onUpload(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragging(true)
  }

  const handleDragLeave = () => setDragging(false)

  return (
    <>
      {/* Upload Zone */}
      <div className="upload-section">
        <div
          className={`upload-zone ${dragging ? 'dragging' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <p>
            📄 Drop a file here or <strong>click to upload</strong>
            <br />
            <small style={{ color: 'var(--text-muted)' }}>
              Supports PDF, Markdown, and text files
            </small>
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.md,.markdown,.txt"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
        {uploadStatus && (
          <div className={`upload-status ${uploadStatus.type}`}>
            {uploadStatus.type === 'success' ? '✓' : '✗'} {uploadStatus.message}
          </div>
        )}
      </div>

      {/* Query Input */}
      <div className="query-section">
        <form className="query-form" onSubmit={handleSubmit}>
          <input
            id="query-input"
            className="query-input"
            type="text"
            placeholder="Ask a question about your documents..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
            autoFocus
          />
          <button
            id="query-submit"
            className="query-btn"
            type="submit"
            disabled={loading || !query.trim()}
          >
            {loading ? 'Running...' : 'Query'}
          </button>
        </form>
      </div>
    </>
  )
}

export default QueryInput
