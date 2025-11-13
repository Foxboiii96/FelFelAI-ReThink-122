import { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000';

export default function App() {
  const [prompt, setPrompt] = useState('Explain how to add 3 and 4.');
  const [completion, setCompletion] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed) {
      setError('Please enter a prompt for the model.');
      return;
    }

    setIsLoading(true);
    setError('');
    setCompletion('');

    try {
      const response = await fetch(`${API_BASE_URL}/api/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: trimmed }),
      });

      if (!response.ok) {
        const details = await response.json().catch(() => ({}));
        throw new Error(details.detail || 'The server responded with an error.');
      }

      const data = await response.json();
      setCompletion(data.completion);
    } catch (err) {
      setError(err.message || 'Something went wrong while contacting the API.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Mini Reasoning LLM</h1>
        <p>Experiment with a tiny PyTorch-powered model that can reason about numbers.</p>
      </header>

      <main>
        <form className="prompt-form" onSubmit={handleSubmit}>
          <label htmlFor="prompt" className="prompt-label">
            Prompt
          </label>
          <textarea
            id="prompt"
            name="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Ask the model to solve a small reasoning task..."
            rows={6}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Thinking…' : 'Generate'}
          </button>
        </form>

        {error && <div className="alert error">{error}</div>}

        {completion && (
          <section className="result">
            <h2>Model Response</h2>
            <p>{completion}</p>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          Backend URL: <code>{API_BASE_URL}</code>
        </p>
      </footer>
    </div>
  );
}
