import { useState } from 'react';
import './styles/App.css';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function App() {
  const [prompt, setPrompt] = useState('');
  const [completion, setCompletion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!prompt.trim()) {
      setError('Please enter a prompt before submitting.');
      return;
    }

    setError('');
    setLoading(true);
    setCompletion('');

    try {
      const response = await fetch(`${apiBaseUrl}/api/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const message = payload?.detail || 'The server returned an error.';
        throw new Error(message);
      }

      const data = await response.json();
      setCompletion(data.completion);
    } catch (fetchError) {
      setError(fetchError.message || 'Something went wrong while contacting the API.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app__header">
        <h1>Mini Reasoning LLM</h1>
        <p>
          Ask the model to add two numbers, check if a number is even or odd, or reverse a word.
          It will explain its reasoning step by step.
        </p>
      </header>

      <main className="app__content">
        <form className="app__form" onSubmit={handleSubmit}>
          <label htmlFor="prompt" className="app__label">
            Prompt
          </label>
          <textarea
            id="prompt"
            className="app__textarea"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="e.g. Can you explain 2 plus 3 step by step?"
            rows={5}
          />
          <button className="app__button" type="submit" disabled={loading}>
            {loading ? 'Thinking...' : 'Generate Reasoning'}
          </button>
        </form>

        {error && <div className="app__error">{error}</div>}

        {completion && (
          <section className="app__result">
            <h2>Model Response</h2>
            <pre>{completion}</pre>
          </section>
        )}
      </main>

      <footer className="app__footer">
        API base URL: <code>{apiBaseUrl}</code>
      </footer>
    </div>
  );
}
