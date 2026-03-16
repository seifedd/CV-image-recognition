import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      processFile(selectedFile)
    }
  }

  const processFile = (selectedFile) => {
    // Basic security validation: check if it's an image
    if (!selectedFile.type.startsWith('image/')) {
      setError('Please upload a valid image file.')
      return
    }
    setError(null)
    setResult(null)
    setFile(selectedFile)
    
    // Create preview URL to display
    const objectUrl = URL.createObjectURL(selectedFile)
    setPreview(objectUrl)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0])
    }
  }

  const handleSubmit = async () => {
    if (!file) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Connect to the backend
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Server responded with an error.')
      }

      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }

      setResult(data.prediction)
    } catch (err) {
      setError(err.message || 'Failed to analyze the image. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Vision Intelligence</h1>
        <p>Advanced K-NN Image Classification</p>
      </header>

      <main className="main-card">
        {!preview ? (
          <div 
            className="dropzone"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange} 
              accept="image/*" 
              style={{ display: 'none' }} 
            />
            <svg className="upload-icon" fill="none" strokeWidth="1.5" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
            </svg>
            <h3 className="dropzone-text">Click or drag image to upload</h3>
            <p className="dropzone-subtext">Supports JPG, PNG (Max 5MB)</p>
          </div>
        ) : (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="image-preview" />
            
            <div className="actions">
              <button 
                className="btn btn-secondary" 
                onClick={handleReset}
                disabled={loading}
              >
                Cancel
              </button>
              <button 
                className="btn btn-primary" 
                onClick={handleSubmit} 
                disabled={loading || result}
              >
                {loading ? <span className="loader"></span> : 'Analyze Image'}
              </button>
            </div>
            
            {error && (
              <div className="result-box error">
                <div className="result-label">Error</div>
                <div className="result-value" style={{ fontSize: '1rem', fontWeight: 500 }}>{error}</div>
              </div>
            )}
            
            {result && (
              <div className="result-box">
                <div className="result-label">Detected Class</div>
                <div className="result-value">{result}</div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

export default App
